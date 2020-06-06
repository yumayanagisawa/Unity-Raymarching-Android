Shader "Unlit/Metaballs"
{
    Properties
    {
        _MainTex ("Texture", 2D) = "white" {}
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
			uniform float4x4 _ObjMatrix_InvModel;

			uniform float _DrawDistance;


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

			// metaballs
			float sphere(float3 pos)
			{
				return length(pos) - .3;
			}

			float blob5(float d1, float d2, float d3, float d4, float d5)
			{
				float k = 2.0;
				return -log(exp(-k * d1) + exp(-k * d2) + exp(-k * d3) + exp(-k * d4) + exp(-k * d5)) / k;
			}

			float scene(float3 pos, float Time)
			{
				float t = Time;

				float ec = .9;
				float s1 = sphere(pos - ec * float3(cos(t*1.1), cos(t*1.3), cos(t*1.7)));
				float s2 = sphere(pos + ec * float3(cos(t*0.7), cos(t*1.9), cos(t*2.)));
				float s3 = sphere(pos + ec * float3(cos(t*0.3), cos(t*1.2), sin(t*1.1)));
				float s4 = sphere(pos + ec * float3(sin(t*1.3), sin(t*1.7), sin(t*0.7)));
				float s5 = sphere(pos + ec * float3(sin(t*2.3), sin(t*1.9), sin(t*.9)));

				return blob5(s1, s2, s3, s4, s5);
			}
			
			// This is the distance field function.  The distance field represents the closest distance to the surface
			// of any object we put in the scene.  If the given point (point p) is inside of an object, we return a
			// negative answer.
			// return.x: result of distance field
			// return.y: material data for closest object
			float map(float3 p) {
				// Apply inverse model matrix via C#
				float3 dynamic_p = mul(_ObjMatrix_InvModel, float4(p, 1)).xyz;
				
				return scene(dynamic_p, _Time.y);
			}

			float3 calcNormal(in float3 pos)
			{
				const float2 eps = float2(0.001, 0.0);
				// The idea here is to find the "gradient" of the distance field at pos
				// Remember, the distance field is not boolean - even if you are inside an object
				// the number is negative, so this calculation still works.
				// Essentially you are approximating the derivative of the distance field at this point.
				float3 nor = float3(
					map(pos + eps.xyy) - map(pos - eps.xyy),
					map(pos + eps.yxy) - map(pos - eps.yxy),
					map(pos + eps.yyx) - map(pos - eps.yyx));
				return normalize(nor);
			}

			// Custom function setting colours
			float3 setColour(float3 ro, float3 rd, inout float3 colour, float3 currPos) {
				float3 normal = calcNormal(currPos);
				float ndotl = abs(dot(-rd, normal));
				float rim = pow(1.0 - ndotl, 1.0);
				colour = float3(1.0, 1.0, 1.0);
				colour = lerp(colour, -normal*0.5 + float3(0.5, 0., 0.5), rim + 0.1);
				return colour;
			}

			// Raymarch along given ray
			// ro: ray origin
			// rd: ray direction
			// s: unity depth buffer
			fixed4 raymarch(float3 ro, float3 rd, float s) {
				fixed4 ret = fixed4(0, 0, 0, 0);
				float3 colour = float3(0, 0, 0);
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
					float d = map(p);		// Sample of distance field (see map())

					// If the sample <= 0, we have hit something (see map()).
					if (d.x < 0.001) {
						float3 n = calcNormal(p);
						//float light = dot(-_LightDir.xyz, n);
						// just pick grey
						//ret = fixed4(float3(0.8, 0.8, 0.8) * light, 1);						
						ret = fixed4(setColour(ro, rd, colour, p), 1.0);
						break;
					}

					// If the sample > 0, we haven't hit anything yet so we should march forward
					// We step forward by distance d, because d is the minimum distance possible to intersect
					// an object (see map()).
					t += d;
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
