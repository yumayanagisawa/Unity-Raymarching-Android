// ref: https://www.shadertoy.com/view/lljSDW
Shader "Unlit/Fluid"
{
    Properties
    {
        _MainTex ("Texture", 2D) = "white" {}
		iChannel0("CubeTex", CUBE) = "white" {}
    }
    SubShader
    {
		// No culling or depth
		Cull Off ZWrite Off ZTest Always

        Pass
        {
            CGPROGRAM
            #pragma vertex vert
            #pragma fragment frag
            // make fog work
            #pragma multi_compile_fog

            #include "UnityCG.cginc"
			static const int STEPS = 128;
			static const float INTERSECTION_PRECISION = 0.001;

			uniform sampler2D _CameraDepthTexture;
			// These are are set by our script (see RaymarchGeneric.cs)
			uniform sampler2D _MainTex;
			uniform float4 _MainTex_TexelSize;

			uniform float4x4 _CameraInvViewMatrix;
			uniform float4x4 _FrustumCornersES;
			uniform float4 _CameraWS;

			uniform float3 _LightDir;
			uniform float4x4 _MatTorus_InvModel;

			uniform float _DrawDistance;

			// cubemap
			samplerCUBE iChannel0;

            struct appdata
            {
                float4 vertex : POSITION;
                float2 uv : TEXCOORD0;
            };
			
			struct v2f
			{
				float4 pos : SV_POSITION;
				float2 uv : TEXCOORD0;
				float3 ray : TEXCOORD1;
			};

            //float4 _MainTex_ST; perhaps unnecessary

            v2f vert (appdata v)
            {
				v2f o;

				// Index passed via custom blit function in RaymarchGeneric.cs
				half index = v.vertex.z;
				v.vertex.z = 0.1;

				o.pos = UnityObjectToClipPos(v.vertex);
				o.uv = v.uv.xy;

				#if UNITY_UV_STARTS_AT_TOP
				if (_MainTex_TexelSize.y < 0)
					o.uv.y = 1 - o.uv.y;
				#endif

				// Get the eyespace view ray (normalized)
				o.ray = _FrustumCornersES[(int)index].xyz;
				// Dividing by z "normalizes" it in the z axis
				// Therefore multiplying the ray by some number i gives the viewspace position
				// of the point on the ray with [viewspace z]=i
				o.ray /= abs(o.ray.z);

				// Transform the ray from eyespace to worldspace
				o.ray = mul(_CameraInvViewMatrix, o.ray);

				return o;
            }

			float udRoundBox(float3 p, float3 b, float r)
			{
				float undulate = 6. * cos(_Time.y * 0.2);
				//float radius = r + .185 * (sin(p.x * undulate) + sin(p.y * undulate + 2.5*_Time.y));
				float radius = r + .085 * (sin(p.x * undulate) + sin(p.y * undulate + 2.5*_Time.y));
				return length(max(abs(p) - b, 0.0)) - r * radius;
			}

			float sdSphere(float3 p, float s)
			{
				return length(p) - s;
			}

			// union
			float2 opU(float2 d1, float2 d2)
			{
				return d1.x < d2.x ? d1 : d2;
			}

			float smin(float a, float b, float k)
			{
				float res = exp(-k * a) + exp(-k * b);
				return -log(res) / k;
			}

			// iq's noise func
			float hash(float n) { return frac(sin(n)*753.5453123); }
			float noise(in float3 x)
			{
				float3 p = floor(x);
				float3 f = frac(x);
				f = f * f*(3.0 - 2.0*f);

				float n = p.x + p.y*157.0 + 113.0*p.z;
				return lerp(lerp(lerp(hash(n + 0.0), hash(n + 1.0), f.x),
					lerp(hash(n + 157.0), hash(n + 158.0), f.x), f.y),
					lerp(lerp(hash(n + 113.0), hash(n + 114.0), f.x),
						lerp(hash(n + 270.0), hash(n + 271.0), f.x), f.y), f.z);
			}

			// This is the distance field function.  The distance field represents the closest distance to the surface
			// of any object we put in the scene.  If the given point (point p) is inside of an object, we return a
			// negative answer.
			// return.x: result of distance field
			// return.y: material data for closest object
			float2 map(float3 p) {
				// Apply inverse model matrix to point when sampling torus
				// This allows for more complex transformations/animation on the torus

				// get torus world position
				float3 torus_p = mul(_MatTorus_InvModel, float4(p, 1)).xyz;

				float2 blobA = float2(udRoundBox(p - float3(0.0, 0.0, 0.0), float3(0.0, 0.0, 0.0), .5), 0.5);
				float2 blob = float2(udRoundBox(torus_p - float3(0.0, 0.0, 0.0), float3(0.0, 0.0, 0.0), .5), 0.5);

				return blobA;// opU(blob, blobA);// d;// ret;
			}

			// outside
			float2 map_(float3 pos) {
				//pos = mul(_MatTorus_InvModel, float4(pos, 1)).xyz;
				//float2 d2 = float2(pos.y + 2.0, 2.0); plane
				float size = 1.;// fmod(_Time.y*0.125, 3) + 0.25; // 1.75 initially
				//float t1 = sdSphere(pos, size) + noise(pos * 1.0 + _Time.y * 0.75);
				float t1 = sdSphere(pos, size) + noise(pos * 0.3 + _Time.y * .35);

				//t1 = smin(t1, sdSphere(pos + float3(1.8, 2.0, 0.0), 0.2), 2.0);
				//t1 = smin(t1, sdSphere(pos + float3(-1.8, 2.0, -1.0), 0.2), 2.0);

				return float2(t1, 1.0);
			}

			// inside
			float2 map2(float3 pos) {
				//pos = mul(_MatTorus_InvModel, float4(pos, 1)).xyz;
				//float sphere = distSphere(pos, 1.0) + noise(pos * 1.2 + float3(-0.3) + iTime*0.2);
				float size = fmod(_Time.y*0.0625, 1.5); // 1.75 initially
				float sphere = sdSphere(pos, size);

				//sphere = smin(sphere, sdSphere(pos + float3(-0.4, 0.0, -1.0), 0.304), 15.0);
				//sphere = smin(sphere, sdSphere(pos + float3(-0.5, -0.75, 0.0), 0.305), 50.0);
				//sphere = smin(sphere, sdSphere(pos + float3(0.5, 0.7, 0.5), 0.31), 15.0);

				return float2(sphere, 1.0);
			}

			float3 calcNormal(in float3 pos)
			{
				const float2 eps = float2(0.001, 0.0);
				// The idea here is to find the "gradient" of the distance field at pos
				// Remember, the distance field is not boolean - even if you are inside an object
				// the number is negative, so this calculation still works.
				// Essentially you are approximating the derivative of the distance field at this point.
				float3 nor = float3(
					map(pos + eps.xyy).x - map(pos - eps.xyy).x,
					map(pos + eps.yxy).x - map(pos - eps.yxy).x,
					map(pos + eps.yyx).x - map(pos - eps.yyx).x);
				return normalize(nor);
			}
			
			float3 calcNormal2(in float3 pos) {

				float3 eps = float3(0.001, 0.0, 0.0);
				float3 nor = float3(
					map2(pos + eps.xyy).x - map2(pos - eps.xyy).x,
					map2(pos + eps.yxy).x - map2(pos - eps.yxy).x,
					map2(pos + eps.yyx).x - map2(pos - eps.yyx).x);
				return normalize(nor);
			}

			float3 setInsideCol(float3 ro, float3 rd, inout float3 colour, float3 p) {
				float3 normal = calcNormal2(p);
				float ndotl = abs(dot(-rd, normal));
				float rim = pow(1.0 - ndotl, 3.0);
				colour = lerp(refract(normal, rd, 0.5)*0.5 + float3(0.5, 0.5, 0.5), colour, rim);
				float4 pReflect = float4(reflect(rd, normal), 0);
				colour += texCUBElod(iChannel0, pReflect).xyz * 0.05;
				return colour;
			}

			void marchForInsideCol(float3 ro, float3 rd, inout float3 colour) {
				float t, d = 0.;
				for (int i = 0; i < STEPS; ++i)
				{
					float3 currPos = ro + rd * t;
					d = map2(currPos).x;
					if (d < INTERSECTION_PRECISION)
					{
						setInsideCol(ro, rd, colour, currPos);
						break;
					}
					t += d;
				}
			}

			// Custom function setting colours
			float3 setColour(float3 ro, float3 rd, inout float3 colour, float3 currPos) {
				float3 normal = calcNormal(currPos);
				
				// ink-like noise
				float3 normal_distorted = calcNormal(currPos + noise(currPos*1.5 + float3(0.0, 0.0, sin(_Time.y*0.75))));
				float ndotl_distorted = abs(dot(-rd, normal_distorted));
				float rim_distorted = pow(1.0 - ndotl_distorted, 6.0);
				
				float4 pRefract = float4(refract(rd, normal, .95), 0);
				float4 pReflect = float4(reflect(rd, normal), 0);
				colour = texCUBElod(iChannel0, pRefract).xyz;
				colour += texCUBElod(iChannel0, pReflect).xyz * 0.15;
				colour *= 0.87;
				colour = lerp(colour, normal*0.5 + float3(0.5, 0.5, 0.5), rim_distorted + 0.1);
				

				// inside
				//marchForInsideCol(currPos, refract(rd, normal, 0.85), colour);
				return colour;
			}

			// Raymarch along given ray
			// ro: ray origin
			// rd: ray direction
			// s: unity depth buffer
			fixed4 raymarch(float3 ro, float3 rd, float s) {
				fixed4 ret = fixed4(0, 0, 0, 0);
				float3 colour = texCUBE(iChannel0, rd).xyz;
				//const int maxstep = 128;// 64;
				float t = 0; // current distance traveled along ray
				for (int i = 0; i < STEPS; ++i) {
					// If we run past the depth buffer, or if we exceed the max draw distance,
					// stop and return nothing (transparent pixel).
					// this way raymarched objects and traditional meshes can coexist.
					if (t >= s || t > _DrawDistance) {
						ret = fixed4(0, 0, 0, 0);
						break;
					}

					float3 p = ro + rd * t; // World space position of sample
					float2 d = map(p);		// Sample of distance field (see map())

					// If the sample <= 0, we have hit something (see map()).
					if (d.x < 0.001) {
						float3 n = calcNormal(p);
						float light = dot(-_LightDir.xyz, n);
						// just pick grey
						//ret = fixed4(float3(0.5, 0.5, 0.5) * light, 1);
						// render color
						
						ret = fixed4(setColour(ro, rd, colour, p), 1.0);
						break;
					}

					// If the sample > 0, we haven't hit anything yet so we should march forward
					// We step forward by distance d, because d is the minimum distance possible to intersect
					// an object (see map()).
					t += d.x;
				}

				return ret;
			}

            fixed4 frag (v2f i) : SV_Target
            {
				// ray direction
				float3 rd = normalize(i.ray.xyz);
				// ray origin (camera position)
				float3 ro = _CameraWS;

				float2 duv = i.uv;
				#if UNITY_UV_STARTS_AT_TOP
				if (_MainTex_TexelSize.y < 0)
					duv.y = 1 - duv.y;
				#endif

				// Convert from depth buffer (eye space) to true distance from camera
				// This is done by multiplying the eyespace depth by the length of the "z-normalized"
				// ray (see vert()).  Think of similar triangles: the view-space z-distance between a point
				// and the camera is proportional to the absolute distance.
				float depth = LinearEyeDepth(tex2D(_CameraDepthTexture, duv).r);
				depth *= length(i.ray);

				fixed3 col = tex2D(_MainTex,i.uv);

				#if defined (DEBUG_PERFORMANCE)
				fixed4 add = raymarch_perftest(ro, rd, depth);
				#else
				fixed4 add = raymarch(ro, rd, depth);
				#endif

				// Returns final color using alpha blending
				return fixed4(col*(1.0 - add.w) + add.xyz * add.w,1.0);
            }
            ENDCG
        }
    }
}
