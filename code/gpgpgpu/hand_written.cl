//#define NATIVE_OPS
//#define NATIVE_DIVIDE
#define TAN(x) tan(x)

#ifndef NATIVE_DIVIDE
#define DIVIDE(a,b) a / b
#else
#define DIVIDE(a,b) native_divide(a,b)
#endif

#ifndef NATIVE_OPS
#define SIN(x) sin(x)
#define SQRT(x) sqrt(x)
#else
//#define TAN(x) native_tan(x) //seems to cause some problems
#define SIN(x) native_sin(x)
#define SQRT(x) native_sqrt(x)
#endif

void symmetric_invert(float m[16], float out[16])
{
    ///[0, 1, 2, 3]
    ///[4, 5, 6, 7]
    ///[8, 9, 10,11]
    ///[12,13,14,15]
    float det = 0;

    out[0] = m[5] * m[10] * m[15] -
             m[5] * m[11] * m[11] -
             m[6] * m[6]  * m[15] +
             m[6] * m[7]  * m[11] +
             m[7] * m[6]  * m[11] -
             m[7] * m[7]  * m[10];

    out[1] = -m[1] * m[10] * m[15] +
              m[1] * m[11] * m[11] +
              m[6] * m[2] * m[15] -
              m[6] * m[3] * m[11] -
              m[7] * m[2] * m[11] +
              m[7] * m[3] * m[10];

    out[5] = m[0] * m[10] * m[15] -
             m[0] * m[11] * m[11] -
             m[2] * m[2] * m[15] +
             m[2] * m[3] * m[11] +
             m[3] * m[2] * m[11] -
             m[3] * m[3] * m[10];


    out[2] = m[1] * m[6] * m[15] -
             m[1] * m[7] * m[11] -
             m[5] * m[2] * m[15] +
             m[5] * m[3] * m[11] +
             m[7] * m[2] * m[7] -
             m[7] * m[3] * m[6];

    out[6] = -m[0] * m[6] * m[15] +
              m[0] * m[7] * m[11] +
              m[1] * m[2] * m[15] -
              m[1] * m[3] * m[11] -
              m[3] * m[2] * m[7] +
              m[3] * m[3] * m[6];

    out[10] = m[0] * m[5] * m[15] -
              m[0] * m[7] * m[7] -
              m[1] * m[1] * m[15] +
              m[1] * m[3] * m[7] +
              m[3] * m[1] * m[7] -
              m[3] * m[3] * m[5];

    out[3] = -m[1] * m[6] * m[11] +
              m[1] * m[7] * m[10] +
              m[5] * m[2] * m[11] -
              m[5] * m[3] * m[10] -
              m[6] * m[2] * m[7] +
              m[6] * m[3] * m[6];

    out[7] = m[0] * m[6] * m[11] -
             m[0] * m[7] * m[10] -
             m[1] * m[2] * m[11] +
             m[1] * m[3] * m[10] +
             m[2] * m[2] * m[7] -
             m[2] * m[3] * m[6];

    out[11] = -m[0] * m[5] * m[11] +
               m[0] * m[7] * m[6] +
               m[1] * m[1] * m[11] -
               m[1] * m[3] * m[6] -
               m[2] * m[1] * m[7] +
               m[2] * m[3] * m[5];

    out[15] = m[0] * m[5] * m[10] -
              m[0] * m[6] * m[6] -
              m[1] * m[1] * m[10] +
              m[1] * m[2] * m[6] +
              m[2] * m[1] * m[6] -
              m[2] * m[2] * m[5];

    out[4] = out[1];
    out[8] = out[2];
    out[12] = out[3];
    out[9] = out[6];
    out[13] = out[7];
    out[14] = out[11];

    det = m[0] * out[0] + m[1] * out[4] + m[2] * out[8] + m[3] * out[12];

    det = DIVIDE(1.0f,det);

    for(int y=0; y < 4; y++)
    {
        for(int x=0; x < 4; x++)
        {
            out[y * 4 + x] *= det;
        }
    }
}

void schwarzschild_metric(float4 position, float metric[16])
{
    float rs = 1;
    float r = position.y;
    float theta = position.z;

    metric[0 * 4 + 0] = -(1-DIVIDE(rs,r));
    metric[1 * 4 + 1] = DIVIDE(1,(1-DIVIDE(rs,r)));
    metric[2 * 4 + 2] = r*r;
    metric[3 * 4 + 3] = r*r * SIN(theta)*SIN(theta);
}

struct tetrad
{
    float4 v[4];
};

struct tetrad calculate_schwarzschild_tetrad(float4 position)
{
    float rs = 1;
    float r = position.y;
    float theta = position.z;

    struct tetrad result;
    result.v[0] = (float4)(1/SQRT(1 - rs/r), 0, 0, 0);
    result.v[1] = (float4)(0, SQRT(1 - rs/r), 0, 0);
    result.v[2] = (float4)(0, 0, 1/r, 0);
    result.v[3] = (float4)(0, 0, 0, 1/(r * SIN(theta)));

    return result;
};

float3 get_ray_through_pixel(int sx, int sy, int screen_width, int screen_height, float fov_degrees)
{
    float fov_rad = (fov_degrees / 360.f) * 2 * (float)M_PI;
    float f_stop = (float)(screen_width/2) / TAN(fov_rad/2);

    float3 pixel_direction = (float3)(sx - screen_width/2, sy - screen_height/2, f_stop);

    return normalize(pixel_direction);
}

struct geodesic
{
    float4 position;
    float4 velocity;
};

struct geodesic make_lightlike_geodesic(float4 position, float3 direction, struct tetrad* tetrads)
{
    struct geodesic g;
    g.position = position;
    g.velocity = tetrads->v[0] * -1 //Flipped time component, we're tracing backwards in time
               + tetrads->v[1] * direction[0]
               + tetrads->v[2] * direction[1]
               + tetrads->v[3] * direction[2];

    return g;
}

//it isn't possible to use generics in OpenCL, so I use a hardcoded differentiator
void diff_schwarzs(float4 position, int direction, float dMetric[16])
{
    float4 p_up = position;
    float4 p_lo = position;

    float h = 0.00001f;

    //note that this eliminated by the compiler
    //the lack of ability to access a float4 as an array is one of the major complaints i have in OpenCL
    if(direction == 0)
    {
        p_up.x += h;
        p_lo.x -= h;
    }
    if(direction == 1)
    {
        p_up.y += h;
        p_lo.y -= h;
    }
    if(direction == 2)
    {
        p_up.z += h;
        p_lo.z -= h;
    }
    if(direction == 3)
    {
        p_up.w += h;
        p_lo.w -= h;
    }

    float metric_up[16] = {0};
    schwarzschild_metric(p_up, metric_up);

    float metric_lo[16] = {0};
    schwarzschild_metric(p_lo, metric_lo);

    for(int i=0; i < 16; i++)
    {
        dMetric[i] = (metric_up[i] - metric_lo[i]) * (1/(2*h));
    }
}

void calculate_christoff2(float4 position, float christoff2[64]) {
    float metric[16] = {0};
    schwarzschild_metric(position, metric);

    float metric_inverse[16];
    symmetric_invert(metric, metric_inverse);

    float metric_diff[64];

    for(int i=0; i < 4; i++) {
        float differentiated[16];
        diff_schwarzs(position, i, differentiated);

        for(int j=0; j < 4; j++) {
            for(int k=0; k < 4; k++) {
                metric_diff[i * 16 + j * 4 + k] = differentiated[j * 4 + k];
            }
        }
    }

    for(int mu = 0; mu < 4; mu++)
    {
        for(int al = 0; al < 4; al++)
        {
            for(int be = 0; be < 4; be++)
            {
                float sum = 0;

                for(int sigma = 0; sigma < 4; sigma++)
                {
                    sum += 0.5f * metric_inverse[mu * 4 + sigma] * (metric_diff[be * 16 + sigma * 4 + al] + metric_diff[al * 16 + sigma * 4 + be] - metric_diff[sigma * 16 + al * 4 + be]);
                }

                christoff2[mu * 16 + al * 4 + be] = sum;
            }
        }
    }
}

//use the geodesic equation to get our acceleration
float4 calculate_acceleration_of(float4 X, float4 v) {
    float christoff2[64];
    calculate_christoff2(X, christoff2);

    float acceleration[4];
    float va[4] = {v.x, v.y, v.z, v.w};

    for(int mu = 0; mu < 4; mu++) {
        float sum = 0;

        for(int al = 0; al < 4; al++) {
            for(int be = 0; be < 4; be++) {
                sum += -christoff2[mu * 16 + al * 4 + be] * va[al] * va[be];
            }
        }

        acceleration[mu] = sum;
    }

    return (float4)(acceleration[0], acceleration[1], acceleration[2], acceleration[3]);
}

int integrate(struct geodesic* g)
{
    int result = 2;

    float dt = 0.005f;
    float rs = 1;
    float start_time = g->position.x;

    for(int idx = 0; idx < 1024 * 1024; idx++)
    {
        float4 acceleration = calculate_acceleration_of(g->position, g->velocity);

        g->velocity += acceleration * dt;
        g->position += g->velocity * dt;

        float radius = g->position.y;

        if(radius > 10)
        {
            result = 0;
            break;
        }

        if(radius <= rs + 0.0001f || g->position.x > start_time + 1000)
        {
            result = 1;
            break;
        }

        //these kinds of vector expressions are one of the nicer parts of OpenCL
        if(!any(isfinite(g->position)))
        {
            result = 1;
            break;
        }
    }

    return result;
}

float2 angle_to_tex(float2 angle)
{
    float pi = M_PI;

    float thetaf = fmod(angle.x, 2 * pi);
    float phif = angle.y;

    if(thetaf >= pi)
    {
        phif += pi;
        thetaf -= pi;
    }

    phif = fmod(phif, 2 * pi);

    float sxf = phif / (2 * pi);
    float syf = thetaf / pi;

    sxf += 0.5f;

    return (float2)(sxf, syf);
}

float3 render_pixel(int x, int y, int screen_width, int screen_height, read_only image2d_t background, int background_width, int background_height)
{
    float3 ray_direction = get_ray_through_pixel(x, y, screen_width, screen_height, 90);

    float4 camera_position = (float4)(0, 5, (float)M_PI/2, -(float)M_PI/2);

    struct tetrad tetrads = calculate_schwarzschild_tetrad(camera_position);

    //so, the tetrad vectors give us a basis, that points in the direction t, r, theta, and phi, because schwarzschild is diagonal
    //we'd like the ray to point towards the black hole: this means we make +z point towards -r, +y point towards +theta, and +x point towards +phi
    float3 modified_ray = {-ray_direction.z, ray_direction.y, ray_direction.x};

    struct geodesic my_geodesic = make_lightlike_geodesic(camera_position, modified_ray, &tetrads);

    int result = integrate(&my_geodesic);

    float theta = my_geodesic.position.z;
    float phi = my_geodesic.position.w;

    float2 texture_coordinate = angle_to_tex((float2){theta, phi});

    int tx = (int)(texture_coordinate.x * background_width + background_width) % background_width;
    int ty = (int)(texture_coordinate.y * background_height + background_height) % background_height;

    float3 colour = read_imagef(background, (int2)(tx, ty)).xyz;

    if(result == 2 || result == 1)
    {
        colour = (float3)(0,0,0);
    }

    return colour;
}

__kernel void hand_raytracer(int screen_width, int screen_height, read_only image2d_t background, write_only image2d_t screen, int background_width, int background_height)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if(y >= screen_height)
        return;

    if(x >= screen_width)
        return;

    float3 colour = render_pixel(x, y, screen_width, screen_height, background, background_width, background_height);

    float4 crgba = {colour.x, colour.y, colour.z, 1.f};

    write_imagef(screen, (int2){x, y}, crgba);
}
