/*
 * Eclipse Simulator - an educational OpenGL / GLUT program written in C.
 *
 * Shows the Sun, Earth and Moon, their orbits and shadows, and explains
 * when and why solar and lunar eclipses happen.
 *
 * All lighting is computed on the CPU: for every vertex of the Earth and the
 * Moon we compute how much of the Sun's disc is visible from that point
 * (taking the other body as an occluder).  This gives physically correct
 * umbra and penumbra shading, a dark shadow spot on Earth during solar
 * eclipses and a reddish Moon during total lunar eclipses.
 *
 * NOTE: sizes and distances are NOT to scale (the real Solar System would
 * not fit on the screen).  The Moon's orbital tilt is exaggerated by default
 * to compensate for the compressed distances.
 *
 * Realistic surfaces: NASA images in the textures/ folder are wrapped around
 * the Earth (day map, city lights, clouds) and the Moon; the Sun's surface,
 * its glow and the Milky Way are generated procedurally.  If the textures
 * are missing the program falls back to simple procedural colours.
 * Images are loaded with stb_image.h (public domain, by Sean Barrett).
 *
 * Build (Linux):   gcc -O2 -o eclipse_sim eclipse_sim.c -lglut -lGLU -lGL -lm
 * Build (macOS):   clang -O2 -o eclipse_sim eclipse_sim.c -framework OpenGL -framework GLUT
 * Build (MinGW):   gcc -O2 -o eclipse_sim.exe eclipse_sim.c -lfreeglut -lglu32 -lopengl32
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <math.h>
#ifdef __APPLE__
#define GL_SILENCE_DEPRECATION
#include <GLUT/glut.h>
#else
#include <GL/glut.h>
#endif

#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"
#pragma GCC diagnostic ignored "-Wsign-compare"
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wunused-function"
#pragma GCC diagnostic ignored "-Wtype-limits"
#pragma GCC diagnostic ignored "-Wimplicit-fallthrough"
#pragma GCC diagnostic ignored "-Wmissing-field-initializers"
#endif
#define STB_IMAGE_IMPLEMENTATION
#define STBI_ONLY_JPEG
#define STBI_ONLY_PNG
#include "stb_image.h"
#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#define DEG2RAD (M_PI / 180.0)
#define RAD2DEG (180.0 / M_PI)

/* ------------------------------------------------------------------ */
/* Model constants (educational scale, not real scale)                 */
/* ------------------------------------------------------------------ */
#define SUN_RADIUS     3.0
#define EARTH_RADIUS   1.0
#define MOON_RADIUS    0.3
#define EARTH_ORBIT    60.0
#define MOON_ORBIT     7.0
#define MOON_ECC       0.08      /* exaggerated eccentricity -> total and annular eclipses */
#define EARTH_TILT     23.44

#define YEAR_DAYS      365.256
#define DRACONIC_MONTH 27.2122   /* node to node */
#define NODE_PERIOD    6798.38   /* regression of the nodes, 18.6 years */
#define APSIDE_PERIOD  3232.6    /* rotation of the perigee, 8.85 years */

#define NODE_LON0      40.0
#define PERIGEE_LON0   0.0
#define MOON_U0        60.0

#define SLOW_SPEED     0.25      /* days/s while an eclipse is in progress (auto-slow) */

/* ------------------------------------------------------------------ */
/* Small vector library                                                */
/* ------------------------------------------------------------------ */
typedef struct { double x, y, z; } Vec3;

static Vec3 v3(double x, double y, double z) { Vec3 r; r.x = x; r.y = y; r.z = z; return r; }
static Vec3 vadd(Vec3 a, Vec3 b) { return v3(a.x + b.x, a.y + b.y, a.z + b.z); }
static Vec3 vsub(Vec3 a, Vec3 b) { return v3(a.x - b.x, a.y - b.y, a.z - b.z); }
static Vec3 vscale(Vec3 a, double s) { return v3(a.x * s, a.y * s, a.z * s); }
static double vdot(Vec3 a, Vec3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
static Vec3 vcross(Vec3 a, Vec3 b) { return v3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x); }
static double vlen(Vec3 a) { return sqrt(vdot(a, a)); }
static Vec3 vnorm(Vec3 a) { double l = vlen(a); return l > 1e-12 ? vscale(a, 1.0 / l) : v3(0, 0, 0); }
static double clampd(double v, double lo, double hi) { return v < lo ? lo : (v > hi ? hi : v); }
static void glVertexV(Vec3 p) { glVertex3d(p.x, p.y, p.z); }

/* ------------------------------------------------------------------ */
/* Simulation state                                                    */
/* ------------------------------------------------------------------ */
typedef enum {
    EC_NONE = 0,
    EC_SOLAR_PARTIAL, EC_SOLAR_ANNULAR, EC_SOLAR_TOTAL,
    EC_LUNAR_PENUMBRAL, EC_LUNAR_PARTIAL, EC_LUNAR_TOTAL
} EclipseType;

typedef struct {
    double t;
    Vec3 sun, earth, moon;
    double earthAngle;   /* rad */
    double nodeLon;      /* longitude of ascending node, rad */
    double perigeeLon;   /* rad */
    double moonU;        /* argument of latitude (angle from ascending node), rad */
    double moonDist;
    double incl;         /* rad */
    double ecc;
} World;

typedef struct {
    EclipseType solar;   /* solar eclipse type (or EC_NONE) */
    EclipseType lunar;   /* lunar eclipse type (or EC_NONE) */
    double lunarMag;     /* umbral magnitude */
    double solarCov;     /* fraction of the Sun covered, seen from 'observer' */
    Vec3 observer;       /* point on Earth used for the "Sun seen from Earth" view */
    int hasShadowPoint;  /* Moon's shadow axis hits Earth */
    Vec3 shadowPoint;
    double shadowSpotR;
    double penumbraR;    /* radius of the Moon's penumbra at Earth's surface */
} EclipseInfo;

static double simTime = 0.0;      /* days */
static double speed = 1.0;        /* days per second */
static int paused = 0;
static const double inclValues[] = { 0.0, 5.145, 15.0, 30.0 };
static const char *inclNames[] = {
    "0 deg  (flat orbit: eclipses EVERY month)",
    "5.1 deg  (true value)",
    "15 deg  (exaggerated)",
    "30 deg  (exaggerated to suit this scale)"
};
static int inclIdx = 3;
static int eccOn = 1;
static int showOrbits = 1, showShadows = 1, showLabels = 1, showHelp = 1, showFacts = 0, autoSlow = 1;

static int camMode = 2;
static const char *camNames[] = { "Follow Earth", "Overview (Sun and Earth)", "Side view of Sun-Earth line" };
static double camYaw = 0.0, camPitch = 12.0, camDist = 26.0;
static Vec3 camPos;

static int winW = 1280, winH = 800;
static int mouseBtn = -1, mouseX = 0, mouseY = 0;
static int lastMs = 0;
static GLdouble mvMat[16], prMat[16];
static GLint viewportArr[4];

static World curWorld;
static EclipseInfo curInfo;
static char statusMsg[160] = "";
static int statusMsgUntil = 0;

/* command line testing options */
static const char *shotFile = NULL;
static int shotFrames = 0;

#define NUM_STARS 9000
static float starPos[NUM_STARS][3];
static float starBright[NUM_STARS];
static float starCol[NUM_STARS][3];

/* textures (0 = not available) */
static GLuint texEarth, texNight, texClouds, texOcean, texMoon, texSun, texGlow;
static char texDir[1024] = "textures/";
static Vec3 galPole;   /* axis of the Milky Way band */

/* ------------------------------------------------------------------ */
/* Orbital mechanics (simplified)                                      */
/* ------------------------------------------------------------------ */
/* Convert from "math" ecliptic coordinates (X,Y in plane, Z north) to OpenGL (y up). */
static Vec3 eclToGL(double X, double Y, double Z) { return v3(X, Z, -Y); }

static Vec3 moonOffset(const World *w, double u, double *rOut)
{
    double r = MOON_ORBIT * (1.0 - w->ecc * cos(w->nodeLon + u - w->perigeeLon));
    double cO = cos(w->nodeLon), sO = sin(w->nodeLon);
    double cu = cos(u), su = sin(u), ci = cos(w->incl), si = sin(w->incl);
    double X = r * (cO * cu - sO * su * ci);
    double Y = r * (sO * cu + cO * su * ci);
    double Z = r * su * si;
    if (rOut) *rOut = r;
    return eclToGL(X, Y, Z);
}

static void computeWorld(double t, World *w)
{
    w->t = t;
    w->earthAngle = 2.0 * M_PI * t / YEAR_DAYS;
    w->sun = v3(0, 0, 0);
    w->earth = eclToGL(EARTH_ORBIT * cos(w->earthAngle), EARTH_ORBIT * sin(w->earthAngle), 0.0);
    w->incl = inclValues[inclIdx] * DEG2RAD;
    w->ecc = eccOn ? MOON_ECC : 0.0;
    w->nodeLon = (NODE_LON0 - 360.0 * t / NODE_PERIOD) * DEG2RAD;
    w->perigeeLon = (PERIGEE_LON0 + 360.0 * t / APSIDE_PERIOD) * DEG2RAD;
    w->moonU = (MOON_U0 + 360.0 * t / DRACONIC_MONTH) * DEG2RAD;
    w->moon = vadd(w->earth, moonOffset(w, w->moonU, &w->moonDist));
}

/* Area of intersection of two circles with radii r1, r2 and centre distance d. */
static double circleOverlap(double r1, double r2, double d)
{
    double a1, a2, k, rmin;
    if (d >= r1 + r2) return 0.0;
    rmin = r1 < r2 ? r1 : r2;
    if (d <= fabs(r1 - r2)) return M_PI * rmin * rmin;
    a1 = r1 * r1 * acos(clampd((d * d + r1 * r1 - r2 * r2) / (2.0 * d * r1), -1, 1));
    a2 = r2 * r2 * acos(clampd((d * d + r2 * r2 - r1 * r1) / (2.0 * d * r2), -1, 1));
    k = 0.5 * sqrt(fabs((-d + r1 + r2) * (d + r1 - r2) * (d - r1 + r2) * (d + r1 + r2)));
    return a1 + a2 - k;
}

/* Fraction (0..1) of the solar disc visible from P when 'occ' (radius occR) may block it. */
static double visibleSunFraction(const World *w, Vec3 P, Vec3 occ, double occR)
{
    Vec3 ds = vsub(w->sun, P), dO = vsub(occ, P);
    double lS = vlen(ds), lO = vlen(dO), a, b, c;
    if (lO >= lS || lO <= occR) return 1.0;
    a = asin(clampd(SUN_RADIUS / lS, 0, 1));
    b = asin(clampd(occR / lO, 0, 1));
    c = acos(clampd(vdot(ds, dO) / (lS * lO), -1, 1));
    return 1.0 - circleOverlap(a, b, c) / (M_PI * a * a);
}

static void classify(const World *w, EclipseInfo *e)
{
    const double Rs = SUN_RADIUS, Re = EARTH_RADIUS, Rm = MOON_RADIUS;
    Vec3 se = vsub(w->earth, w->sun), d, em, sm, dm, me;
    double D = vlen(se), x, rho, ru, rp, Dm, xe;

    memset(e, 0, sizeof(*e));

    /* ---- Lunar eclipse: is the Moon inside Earth's shadow cone? ---- */
    d = vscale(se, 1.0 / D);
    em = vsub(w->moon, w->earth);
    x = vdot(em, d);
    if (x > 0) {
        rho = vlen(vsub(em, vscale(d, x)));
        ru = Re - x * (Rs - Re) / D;          /* umbra radius at the Moon's distance */
        rp = Re + x * (Rs + Re) / D;          /* penumbra radius */
        if (rho + Rm <= ru) e->lunar = EC_LUNAR_TOTAL;
        else if (rho - Rm < ru) e->lunar = EC_LUNAR_PARTIAL;
        else if (rho - Rm < rp) e->lunar = EC_LUNAR_PENUMBRAL;
        e->lunarMag = (ru - (rho - Rm)) / (2.0 * Rm);
    }

    /* ---- Solar eclipse: does the Moon's shadow reach the Earth? ---- */
    sm = vsub(w->moon, w->sun);
    Dm = vlen(sm);
    dm = vscale(sm, 1.0 / Dm);
    me = vsub(w->earth, w->moon);
    xe = vdot(me, dm);
    e->observer = vadd(w->earth, vscale(vnorm(vsub(w->sun, w->earth)), Re)); /* sub-solar point */
    if (xe > 0) {
        Vec3 q = vadd(w->moon, vscale(dm, xe));
        double xs;
        rho = vlen(vsub(w->earth, q));
        xs = rho < Re ? xe - sqrt(Re * Re - rho * rho) : xe;
        ru = Rm - xs * (Rs - Rm) / Dm;   /* negative => antumbra (annular) */
        rp = Rm + xs * (Rs + Rm) / Dm;
        e->penumbraR = rp;
        if (rho < Re) {
            e->observer = vadd(w->moon, vscale(dm, xs));
            e->hasShadowPoint = 1;
            e->shadowPoint = e->observer;
            e->shadowSpotR = fabs(ru);
        } else if (rho - Re < rp) {
            e->observer = vadd(w->earth, vscale(vnorm(vsub(q, w->earth)), Re));
        }
        if (rho < Re + fabs(ru)) e->solar = ru > 0 ? EC_SOLAR_TOTAL : EC_SOLAR_ANNULAR;
        else if (rho < Re + rp) e->solar = EC_SOLAR_PARTIAL;
    }
    e->solarCov = 1.0 - visibleSunFraction(w, e->observer, w->moon, Rm);
}

static int isSolar(EclipseType t) { return t == EC_SOLAR_PARTIAL || t == EC_SOLAR_ANNULAR || t == EC_SOLAR_TOTAL; }

/* Moon phase: elongation of the Moon from the Sun, as seen from Earth (0 = new, 180 = full). */
static double moonElongation(const World *w)
{
    Vec3 s = vsub(w->sun, w->earth), m = vsub(w->moon, w->earth);
    double sx = s.x, sy = -s.z, mx = m.x, my = -m.z;
    double a = atan2(sx * my - sy * mx, sx * mx + sy * my) * RAD2DEG;
    if (a < 0) a += 360.0;
    return a;
}

static const char *phaseName(double el)
{
    if (el < 8 || el > 352) return "New Moon";
    if (el < 82) return "Waxing Crescent";
    if (el < 98) return "First Quarter";
    if (el < 172) return "Waxing Gibbous";
    if (el < 188) return "Full Moon";
    if (el < 262) return "Waning Gibbous";
    if (el < 278) return "Last Quarter";
    return "Waning Crescent";
}

/* ------------------------------------------------------------------ */
/* Time control helpers                                                */
/* ------------------------------------------------------------------ */
static void setStatus(const char *msg)
{
    snprintf(statusMsg, sizeof(statusMsg), "%s", msg);
    statusMsgUntil = glutGet(GLUT_ELAPSED_TIME) + 4000;
}

static void setCamera(int mode)
{
    static const double dists[] = { 22.0, 120.0, 26.0 };
    static const double pitches[] = { 25.0, 50.0, 12.0 };
    camMode = mode;
    camDist = dists[mode];
    camPitch = pitches[mode];
    camYaw = 0.0;
}

/* Jump to (just before) the next solar or lunar eclipse. */
static int jumpToNext(int solar)
{
    World w;
    EclipseInfo e;
    double t = simTime;
    int k;
    /* first leave any eclipse of this kind that is already in progress */
    for (k = 0; k < 2000; k++, t += 0.02) {
        computeWorld(t, &w);
        classify(&w, &e);
        if ((solar ? e.solar : e.lunar) == EC_NONE) break;
    }
    for (k = 0; k < 400000; k++, t += 0.02) {
        computeWorld(t, &w);
        classify(&w, &e);
        if ((solar ? e.solar : e.lunar) != EC_NONE) {
            simTime = t - 0.4;
            paused = 0;
            if (speed > 1.0) speed = 1.0;
            setCamera(2);
            camYaw = 20.0;
            camPitch = 12.0;
            camDist = 26.0;
            if (solar) {
                /* aim the camera at the spot where the Moon's shadow falls at mid-eclipse */
                double tm, t0 = t, t1 = t, best = -1.0, tt;
                Vec3 n, sd, d;
                for (tt = t; tt < t + 1.0; tt += 0.01) {
                    computeWorld(tt, &w);
                    classify(&w, &e);
                    if (e.solar == EC_NONE) break;
                    if (e.solarCov > best + 1e-6) { best = e.solarCov; t0 = t1 = tt; }
                    else if (e.solarCov >= best - 1e-6) t1 = tt;
                }
                tm = 0.5 * (t0 + t1);
                computeWorld(tm, &w);
                classify(&w, &e);
                n = vnorm(vsub(e.observer, w.earth));
                sd = vnorm(vsub(w.sun, w.earth));
                d = vnorm(vadd(vscale(n, 2.0), sd));
                camYaw = (atan2(d.x, d.z) - w.earthAngle) / DEG2RAD;
                camYaw = fmod(camYaw + 540.0, 360.0) - 180.0;
                camPitch = asin(clampd(d.y, -1.0, 1.0)) / DEG2RAD + 8.0;
                camDist = 9.0;
            }
            setStatus(solar ? "Jumped to the next SOLAR eclipse (New Moon near a node)"
                            : "Jumped to the next LUNAR eclipse (Full Moon near a node)");
            return 1;
        }
    }
    setStatus("No eclipse found in the next 20 years with these settings");
    return 0;
}

static const char *eclipseName(EclipseType t);

/* Print a table of all eclipses in the next 'years' years (useful for lesson planning). */
static void listEclipses(double years)
{
    World w;
    EclipseInfo e;
    double t, start[2] = { 0, 0 };
    int active[2] = { 0, 0 }, best[2] = { 0, 0 }, k, count = 0;
    printf("Eclipses in the next %.0f years (tilt %s, %s orbit):\n", years, inclNames[inclIdx],
           eccOn ? "elliptical" : "circular");
    printf("  %-26s %10s %10s\n", "type", "start day", "end day");
    for (t = simTime; t < simTime + years * YEAR_DAYS; t += 0.02) {
        computeWorld(t, &w);
        classify(&w, &e);
        for (k = 0; k < 2; k++) {
            EclipseType ty = k == 0 ? e.solar : e.lunar;
            if (ty != EC_NONE) {
                if (!active[k]) { active[k] = 1; start[k] = t; best[k] = (int)ty; }
                if ((int)ty > best[k]) best[k] = (int)ty;
            } else if (active[k]) {
                active[k] = 0;
                count++;
                printf("  %-26s %10.2f %10.2f\n", eclipseName((EclipseType)best[k]), start[k], t);
            }
        }
    }
    printf("%d eclipses found.\n", count);
}

static void resetSim(void)
{
    simTime = 0.0; speed = 1.0; paused = 0;
    inclIdx = 3; eccOn = 1;
    showOrbits = showShadows = showLabels = showHelp = autoSlow = 1;
    showFacts = 0;
    setCamera(2);
    setStatus("Simulation reset");
}

/* ------------------------------------------------------------------ */
/* Text / 2D helpers                                                   */
/* ------------------------------------------------------------------ */
static void drawText(float x, float y, void *font, const char *s)
{
    glRasterPos2f(x, y);
    while (*s) glutBitmapCharacter(font, *s++);
}

static void drawTextf(float x, float y, void *font, const char *fmt, ...)
{
    char buf[512];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);
    drawText(x, y, font, buf);
}

static int textWidth(void *font, const char *s)
{
    int w = 0;
    while (*s) w += glutBitmapWidth(font, *s++);
    return w;
}

static void fillRect(float x, float y, float w, float h, float r, float g, float b, float a)
{
    glColor4f(r, g, b, a);
    glBegin(GL_QUADS);
    glVertex2f(x, y); glVertex2f(x + w, y); glVertex2f(x + w, y + h); glVertex2f(x, y + h);
    glEnd();
}

static void strokeRect(float x, float y, float w, float h, float r, float g, float b, float a)
{
    glColor4f(r, g, b, a);
    glBegin(GL_LINE_LOOP);
    glVertex2f(x, y); glVertex2f(x + w, y); glVertex2f(x + w, y + h); glVertex2f(x, y + h);
    glEnd();
}

static void fillCircle2D(float cx, float cy, float r, int seg)
{
    int i;
    glBegin(GL_TRIANGLE_FAN);
    glVertex2f(cx, cy);
    for (i = 0; i <= seg; i++) {
        double a = 2.0 * M_PI * i / seg;
        glVertex2f(cx + r * (float)cos(a), cy + r * (float)sin(a));
    }
    glEnd();
}

static void strokeCircle2D(float cx, float cy, float r, int seg)
{
    int i;
    glBegin(GL_LINE_LOOP);
    for (i = 0; i < seg; i++) {
        double a = 2.0 * M_PI * i / seg;
        glVertex2f(cx + r * (float)cos(a), cy + r * (float)sin(a));
    }
    glEnd();
}

/* ------------------------------------------------------------------ */
/* Surface colouring (CPU lighting with eclipse shadows)               */
/* ------------------------------------------------------------------ */
typedef void (*ColorFn)(const World *w, Vec3 P, Vec3 nWorld, Vec3 nBody, float rgb[3]);

static void earthColor(const World *w, Vec3 P, Vec3 n, Vec3 nb, float rgb[3])
{
    double lat = asin(clampd(nb.y, -1, 1));
    double lon = atan2(-nb.z, nb.x);
    double v = sin(2.0 * lon + 0.5) * cos(1.5 * lat)
             + 0.6 * sin(3.0 * lon + 1.3) * sin(2.0 * lat + 0.4)
             + 0.3 * cos(5.0 * lon - 0.7) * cos(4.0 * lat);
    double base[3], lam, f, light;
    Vec3 toSun = vnorm(vsub(w->sun, P));
    if (fabs(lat) > 70.0 * DEG2RAD) { base[0] = 0.92; base[1] = 0.95; base[2] = 1.0; }
    else if (v > 0.45) {
        double dry = clampd((v - 0.45) * 1.5, 0, 1);
        base[0] = 0.20 + 0.45 * dry; base[1] = 0.55 - 0.1 * dry; base[2] = 0.18 + 0.05 * dry;
    } else { base[0] = 0.08; base[1] = 0.25; base[2] = 0.65; }

    lam = vdot(n, toSun);
    f = lam > 0 ? visibleSunFraction(w, P, w->moon, MOON_RADIUS) : 1.0;
    light = 0.05 + 0.95 * (lam > 0 ? lam : 0.0) * f;
    rgb[0] = (float)(base[0] * light);
    rgb[1] = (float)(base[1] * light);
    rgb[2] = (float)(base[2] * light);
    /* thin bluish atmosphere at the day-side limb */
    if (lam > -0.1) {
        double rim = 1.0 - fabs(vdot(n, vnorm(vsub(camPos, P))));
        rim = rim * rim * rim * 0.5 * clampd(lam + 0.1, 0, 1) * f;
        rgb[0] += (float)(0.3 * rim); rgb[1] += (float)(0.5 * rim); rgb[2] += (float)(1.0 * rim);
    }
}

static void moonColor(const World *w, Vec3 P, Vec3 n, Vec3 nb, float rgb[3])
{
    /* maria (dark patches) on the side that always faces the Earth */
    static const double maria[][3] = {
        { -0.30, 0.35, 0.25 }, { 0.22, 0.30, 0.20 }, { 0.08, -0.08, 0.17 },
        { -0.45, -0.12, 0.20 }, { 0.42, -0.35, 0.12 }, { -0.10, 0.62, 0.12 }
    };
    Vec3 view = vnorm(vsub(w->moon, w->earth));       /* Earth -> Moon */
    Vec3 right = vnorm(vcross(view, v3(0, 1, 0)));
    Vec3 up = vcross(right, view);
    double nr = vdot(n, right), nu = vdot(n, up), nf = -vdot(n, view);
    double albedo = texMoon ? 1.0 : 0.80, lam, f, light, red;
    Vec3 toSun = vnorm(vsub(w->sun, P));
    int i;
    (void)nb;
    if (nf > 0 && !texMoon) {
        for (i = 0; i < 6; i++) {
            double dx = nr - maria[i][0], dy = nu - maria[i][1];
            double dd = sqrt(dx * dx + dy * dy);
            if (dd < maria[i][2]) albedo -= 0.22 * (1.0 - (dd / maria[i][2]) * (dd / maria[i][2]));
        }
    }
    if (albedo < 0.5) albedo = 0.5;
    lam = vdot(n, toSun);
    if (lam < 0) lam = 0;
    f = lam > 0 ? visibleSunFraction(w, P, w->earth, EARTH_RADIUS) : 1.0;
    light = 0.015 + 0.985 * lam * f;
    /* sunlight refracted by Earth's atmosphere lights the umbra in red */
    red = lam * pow(1.0 - f, 4.0) * 0.40;
    rgb[0] = (float)(albedo * light + 0.80 * red);
    rgb[1] = (float)(albedo * light + 0.20 * red);
    rgb[2] = (float)(albedo * light * 0.97 + 0.07 * red);
}

static void sunColor(const World *w, Vec3 P, Vec3 n, Vec3 nb, float rgb[3])
{
    double mu = fabs(vdot(n, vnorm(vsub(camPos, P))));
    double ld = 0.55 + 0.45 * mu;   /* limb darkening */
    (void)w; (void)nb;
    if (texSun) {
        ld = 0.40 + 0.60 * pow(mu, 0.5);
        rgb[0] = (float)clampd(ld * 1.2, 0, 1);
        rgb[1] = (float)clampd(ld * 1.05, 0, 1);
        rgb[2] = (float)clampd(ld * 0.95, 0, 1);
        return;
    }
    rgb[0] = (float)(1.0 * ld + 0.1);
    rgb[1] = (float)(0.82 * ld + 0.05);
    rgb[2] = (float)(0.35 * ld);
}

/* Sphere with per-vertex colour; body normal is rotated by 'spin' around y, then tilted around z. */
static void drawSphereColored(const World *w, Vec3 c, double r, int slices, int stacks,
                              double spin, double tilt, ColorFn fn, GLuint tex)
{
    int i, j, k;
    double ct = cos(tilt), st = sin(tilt);
    if (tex) {
        glEnable(GL_TEXTURE_2D);
        glBindTexture(GL_TEXTURE_2D, tex);
        glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE);
    }
    for (i = 0; i < stacks; i++) {
        glBegin(GL_TRIANGLE_STRIP);
        for (j = 0; j <= slices; j++) {
            for (k = 1; k >= 0; k--) {
                double lat = -M_PI / 2 + M_PI * (i + k) / stacks;
                double lon = 2.0 * M_PI * j / slices;
                Vec3 nb = v3(cos(lat) * cos(lon), sin(lat), -cos(lat) * sin(lon));
                double ls = lon + spin;
                Vec3 ns = v3(cos(lat) * cos(ls), sin(lat), -cos(lat) * sin(ls));
                Vec3 nw = v3(ns.x * ct - ns.y * st, ns.x * st + ns.y * ct, ns.z);
                Vec3 P = vadd(c, vscale(nw, r));
                float rgb[3];
                fn(w, P, nw, nb, rgb);
                glColor3fv(rgb);
                glTexCoord2d(0.5 + lon / (2.0 * M_PI), 0.5 - lat / M_PI);
                glVertexV(P);
            }
        }
        glEnd();
    }
    if (tex) glDisable(GL_TEXTURE_2D);
}

/* Rotation that keeps the Moon's near side (map longitude 0) facing the Earth. */
static double moonSpin(const World *w)
{
    Vec3 d = vsub(w->earth, w->moon);
    return atan2(-d.z, d.x);
}

/* ---- Earth: cached mesh with per-vertex sunlight, drawn in several layers ---- */
#define E_SL 180
#define E_ST 90
typedef struct { Vec3 n, nb; double u, v, lam, f; } EVert;
static EVert eMesh[(E_ST + 1) * (E_SL + 1)];
static const World *passW;
typedef void (*EVertFn)(const EVert *ev, Vec3 P, float rgba[4]);

static void computeEarthMesh(const World *w)
{
    double spin = 2.0 * M_PI * w->t, tilt = EARTH_TILT * DEG2RAD, ct = cos(tilt), st = sin(tilt);
    int i, j;
    for (i = 0; i <= E_ST; i++) {
        double lat = -M_PI / 2 + M_PI * i / E_ST;
        for (j = 0; j <= E_SL; j++) {
            EVert *ev = &eMesh[i * (E_SL + 1) + j];
            double lon = 2.0 * M_PI * j / E_SL, ls = lon + spin;
            Vec3 ns = v3(cos(lat) * cos(ls), sin(lat), -cos(lat) * sin(ls)), P;
            ev->nb = v3(cos(lat) * cos(lon), sin(lat), -cos(lat) * sin(lon));
            ev->n = v3(ns.x * ct - ns.y * st, ns.x * st + ns.y * ct, ns.z);
            P = vadd(w->earth, vscale(ev->n, EARTH_RADIUS));
            ev->lam = vdot(ev->n, vnorm(vsub(w->sun, P)));
            ev->f = ev->lam > -0.05 ? visibleSunFraction(w, P, w->moon, MOON_RADIUS) : 1.0;
            ev->u = 0.5 + lon / (2.0 * M_PI);
            ev->v = 0.5 - lat / M_PI;
        }
    }
}

static void drawEarthLayer(const World *w, double radius, double uOff, GLuint tex, EVertFn fn)
{
    int i, j, k;
    if (tex) {
        glEnable(GL_TEXTURE_2D);
        glBindTexture(GL_TEXTURE_2D, tex);
        glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE);
    }
    for (i = 0; i < E_ST; i++) {
        glBegin(GL_TRIANGLE_STRIP);
        for (j = 0; j <= E_SL; j++) {
            for (k = 1; k >= 0; k--) {
                const EVert *ev = &eMesh[(i + k) * (E_SL + 1) + j];
                Vec3 P = vadd(w->earth, vscale(ev->n, radius));
                float rgba[4];
                fn(ev, P, rgba);
                glColor4fv(rgba);
                glTexCoord2d(ev->u + uOff, ev->v);
                glVertexV(P);
            }
        }
        glEnd();
    }
    if (tex) glDisable(GL_TEXTURE_2D);
}

static double sunlit(const EVert *ev) { return (ev->lam > 0 ? ev->lam : 0.0) * ev->f; }

static void earthDayFn(const EVert *ev, Vec3 P, float c[4])
{
    double L = sunlit(ev), warm;
    if (!texEarth) { earthColor(passW, P, ev->n, ev->nb, c); c[3] = 1.0f; return; }
    /* reddish light near the day/night line (sunrise / sunset) */
    warm = ev->lam > 0 ? clampd(1.0 - ev->lam / 0.25, 0, 1) : 0.0;
    c[0] = (float)clampd(0.012 + 1.05 * L, 0, 1);
    c[1] = (float)clampd(0.012 + 1.05 * L * (1.0 - 0.25 * warm), 0, 1);
    c[2] = (float)clampd(0.020 + 1.05 * L * (1.0 - 0.45 * warm), 0, 1);
    c[3] = 1.0f;
}

static void earthOceanFn(const EVert *ev, Vec3 P, float c[4])
{
    Vec3 hv = vnorm(vadd(vnorm(vsub(passW->sun, P)), vnorm(vsub(camPos, P))));
    double sp = ev->lam > 0 ? pow(clampd(vdot(ev->n, hv), 0, 1), 36.0) * 0.75 * ev->f : 0.0;
    c[0] = (float)(sp * 1.0); c[1] = (float)(sp * 0.95); c[2] = (float)(sp * 0.85); c[3] = 1.0f;
}

static void earthNightFn(const EVert *ev, Vec3 P, float c[4])
{
    double night = 1.0 - clampd(ev->lam * ev->f * 6.0 + 0.1, 0, 1);
    (void)P;
    c[0] = (float)(1.00 * night); c[1] = (float)(0.80 * night); c[2] = (float)(0.55 * night); c[3] = 1.0f;
}

static void earthCloudFn(const EVert *ev, Vec3 P, float c[4])
{
    double L = clampd(0.01 + 1.1 * sunlit(ev), 0, 1);
    (void)P;
    c[0] = c[1] = c[2] = (float)L;
    c[3] = 0.95f;
}

static void earthAtmoFn(const EVert *ev, Vec3 P, float c[4])
{
    double mu = vdot(ev->n, vnorm(vsub(camPos, P))), g, day;
    double t = clampd(mu / 0.30, 0, 1);
    g = mu > 0 ? t * t * (3 - 2 * t) * pow(1.0 - mu, 2.5) : 0.0;
    day = clampd(ev->lam * 1.5 + 0.35, 0, 1) * (0.25 + 0.75 * ev->f);
    c[0] = (float)(0.30 * g * day); c[1] = (float)(0.55 * g * day); c[2] = (float)(1.00 * g * day); c[3] = 1.0f;
}

static void drawEarth(const World *w)
{
    passW = w;
    computeEarthMesh(w);
    glDisable(GL_BLEND);
    drawEarthLayer(w, EARTH_RADIUS, 0.0, texEarth, earthDayFn);
    glEnable(GL_BLEND);
    glDepthMask(GL_FALSE);
    glDepthFunc(GL_LEQUAL);
    if (texOcean) {
        glBlendFunc(GL_SRC_ALPHA, GL_ONE);
        drawEarthLayer(w, EARTH_RADIUS, 0.0, texOcean, earthOceanFn);
    }
    if (texNight) {
        glBlendFunc(GL_ONE, GL_ONE);
        drawEarthLayer(w, EARTH_RADIUS, 0.0, texNight, earthNightFn);
    }
    if (texClouds) {
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        drawEarthLayer(w, EARTH_RADIUS * 1.006, 0.01 * sin(w->t * 0.05), texClouds, earthCloudFn);
    }
    glBlendFunc(GL_ONE, GL_ONE);
    drawEarthLayer(w, EARTH_RADIUS * 1.05, 0.0, 0, earthAtmoFn);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glDepthFunc(GL_LESS);
    glDepthMask(GL_TRUE);
}

/* ------------------------------------------------------------------ */
/* Procedural noise and textures                                       */
/* ------------------------------------------------------------------ */
static double hash3(int x, int y, int z)
{
    unsigned h = (unsigned)x * 73856093u ^ (unsigned)y * 19349663u ^ (unsigned)z * 83492791u;
    h = (h ^ (h >> 13)) * 1274126177u;
    h ^= h >> 16;
    return (h & 0xffffffu) / 16777215.0;
}

static double vnoise(Vec3 p)
{
    int xi = (int)floor(p.x), yi = (int)floor(p.y), zi = (int)floor(p.z);
    double fx = p.x - xi, fy = p.y - yi, fz = p.z - zi, r = 0.0;
    int dx, dy, dz;
    fx = fx * fx * (3 - 2 * fx); fy = fy * fy * (3 - 2 * fy); fz = fz * fz * (3 - 2 * fz);
    for (dz = 0; dz < 2; dz++)
        for (dy = 0; dy < 2; dy++)
            for (dx = 0; dx < 2; dx++)
                r += hash3(xi + dx, yi + dy, zi + dz) * (dx ? fx : 1 - fx) * (dy ? fy : 1 - fy) * (dz ? fz : 1 - fz);
    return r;
}

static double fbm(Vec3 p, int oct)
{
    double sum = 0.0, amp = 0.5, norm = 0.0;
    int i;
    for (i = 0; i < oct; i++) {
        sum += amp * vnoise(p);
        norm += amp;
        p = vscale(p, 2.03);
        amp *= 0.5;
    }
    return sum / norm;
}

static GLuint uploadTex(int w, int h, int comps, GLenum fmt, const unsigned char *px)
{
    GLuint id;
    glGenTextures(1, &id);
    glBindTexture(GL_TEXTURE_2D, id);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
    gluBuild2DMipmaps(GL_TEXTURE_2D, comps, w, h, fmt, GL_UNSIGNED_BYTE, px);
    return id;
}

enum { TEX_RGB, TEX_ALPHA, TEX_OCEAN };

/* Load an equirectangular map from texDir.  TEX_ALPHA turns brightness into
   transparency (clouds), TEX_OCEAN builds a water mask from a day map. */
static GLuint loadTex(const char *file, int mode, double gain)
{
    char path[1200];
    int w, h, n;
    size_t i, np;
    unsigned char *img, *out;
    GLuint id;
    snprintf(path, sizeof(path), "%s%s", texDir, file);
    img = stbi_load(path, &w, &h, &n, 3);
    if (!img) return 0;
    np = (size_t)w * h;
    if (mode == TEX_RGB) {
        for (i = 0; i < np * 3 && gain != 1.0; i++) img[i] = (unsigned char)clampd(img[i] * gain, 0, 255);
        id = uploadTex(w, h, 3, GL_RGB, img);
    } else {
        out = (unsigned char *)malloc(np * 4);
        if (!out) { stbi_image_free(img); return 0; }
        for (i = 0; i < np; i++) {
            int r = img[3 * i], g = img[3 * i + 1], b = img[3 * i + 2];
            double a = mode == TEX_ALPHA ? (r + g + b) / 3.0 * gain
                                         : ((b > r * 1.3 + 8 && b >= g) ? 255.0 : 0.0);
            out[4 * i] = out[4 * i + 1] = out[4 * i + 2] = 255;
            out[4 * i + 3] = (unsigned char)clampd(a, 0, 255);
        }
        id = uploadTex(w, h, 4, GL_RGBA, out);
        free(out);
    }
    stbi_image_free(img);
    return id;
}

/* Sun surface: granulation + a few sunspots, generated on a sphere (no seam). */
static GLuint makeSunTexture(void)
{
    int w = 512, h = 256, x, y;
    unsigned char *px = (unsigned char *)malloc((size_t)w * h * 3);
    GLuint id;
    if (!px) return 0;
    for (y = 0; y < h; y++) {
        double lat = M_PI / 2 - M_PI * (y + 0.5) / h;
        for (x = 0; x < w; x++) {
            double lon = 2.0 * M_PI * (x + 0.5) / w - M_PI;
            Vec3 q = v3(cos(lat) * cos(lon), sin(lat), -cos(lat) * sin(lon));
            double g = fbm(vscale(q, 45.0), 3), big = fbm(vadd(vscale(q, 6.0), v3(7, 3, 1)), 3);
            double c = 0.92 + 0.22 * (g - 0.5);
            unsigned char *o = px + 3 * ((size_t)y * w + x);
            if (big > 0.76 && fabs(lat) < 0.55) c *= clampd(1.0 - (big - 0.76) * 14.0, 0.35, 1.0);
            o[0] = (unsigned char)clampd(255.0 * (1.00 * c + 0.10), 0, 255);
            o[1] = (unsigned char)clampd(255.0 * (0.86 * c + 0.04), 0, 255);
            o[2] = (unsigned char)clampd(255.0 * (0.50 * c), 0, 255);
        }
    }
    id = uploadTex(w, h, 3, GL_RGB, px);
    free(px);
    return id;
}

/* Radial glow sprite (white, brightness in alpha). */
static GLuint makeGlowTexture(void)
{
    int n = 128, x, y;
    unsigned char *px = (unsigned char *)malloc((size_t)n * n * 4);
    GLuint id;
    if (!px) return 0;
    for (y = 0; y < n; y++)
        for (x = 0; x < n; x++) {
            double dx = (x + 0.5) / n * 2 - 1, dy = (y + 0.5) / n * 2 - 1, r = sqrt(dx * dx + dy * dy);
            double a = r < 1.0 ? pow(1.0 - r, 2.2) * 0.55 + exp(-r * 10.0) * 0.45 : 0.0;
            unsigned char *o = px + 4 * ((size_t)y * n + x);
            o[0] = o[1] = o[2] = 255;
            o[3] = (unsigned char)clampd(a * 255.0, 0, 255);
        }
    id = uploadTex(n, n, 4, GL_RGBA, px);
    free(px);
    return id;
}

static void initTextures(const char *argv0)
{
    FILE *fp = fopen("textures/nasa_bluemarble.jpg", "rb");
    if (fp) fclose(fp);
    else {
        const char *s1 = strrchr(argv0, '/'), *s2 = strrchr(argv0, '\\');
        const char *sl = s2 > s1 ? s2 : s1;
        if (sl) snprintf(texDir, sizeof(texDir), "%.*stextures/", (int)(sl - argv0 + 1), argv0);
    }
    texEarth = loadTex("nasa_bluemarble.jpg", TEX_RGB, 1.0);
    texOcean = texEarth ? loadTex("nasa_bluemarble.jpg", TEX_OCEAN, 1.0) : 0;
    texNight = loadTex("nasa_lights.jpg", TEX_RGB, 1.0);
    texClouds = loadTex("nasa_clouds.jpg", TEX_ALPHA, 1.15);
    texMoon = loadTex("nasa_moon.jpg", TEX_RGB, 1.0);
    texSun = makeSunTexture();
    texGlow = makeGlowTexture();
    if (!texEarth || !texMoon)
        fprintf(stderr, "Note: NASA textures not found in '%s' - using simple colours.\n", texDir);
}

/* ------------------------------------------------------------------ */
/* 3D scene pieces                                                     */
/* ------------------------------------------------------------------ */
static void initStars(void)
{
    int i;
    Vec3 e1, e2;
    srand(12345);
    galPole = vnorm(v3(0.35, 0.80, 0.48));
    e1 = vnorm(vcross(galPole, v3(1, 0, 0)));
    e2 = vcross(galPole, e1);
    for (i = 0; i < NUM_STARS; i++) {
        double b = (double)rand() / RAND_MAX, t = (double)rand() / RAND_MAX;
        Vec3 d;
        if (i % 5 < 2) {   /* 40% of the stars crowd into the Milky Way band */
            double lon = 2.0 * M_PI * rand() / RAND_MAX;
            double g = ((double)rand() / RAND_MAX + (double)rand() / RAND_MAX + (double)rand() / RAND_MAX - 1.5) * 0.22;
            d = vadd(vscale(vadd(vscale(e1, cos(lon)), vscale(e2, sin(lon))), cos(g)), vscale(galPole, sin(g)));
        } else {
            double z = 2.0 * rand() / RAND_MAX - 1.0, phi = 2.0 * M_PI * rand() / RAND_MAX, rr = sqrt(1.0 - z * z);
            d = v3(rr * cos(phi), z, rr * sin(phi));
        }
        starPos[i][0] = (float)(900.0 * d.x);
        starPos[i][1] = (float)(900.0 * d.y);
        starPos[i][2] = (float)(900.0 * d.z);
        starBright[i] = (float)(0.12 + 0.88 * pow(b, 4.0));
        if (t < 0.12)      { starCol[i][0] = 0.70f; starCol[i][1] = 0.80f; starCol[i][2] = 1.00f; }
        else if (t < 0.70) { starCol[i][0] = 0.95f; starCol[i][1] = 0.96f; starCol[i][2] = 1.00f; }
        else if (t < 0.90) { starCol[i][0] = 1.00f; starCol[i][1] = 0.93f; starCol[i][2] = 0.78f; }
        else               { starCol[i][0] = 1.00f; starCol[i][1] = 0.75f; starCol[i][2] = 0.55f; }
    }
}

static void drawStars(void)
{
    static const float sizes[3] = { 1.0f, 1.8f, 2.6f };
    static const float lims[4] = { 0.0f, 0.45f, 0.80f, 2.0f };
    Vec3 e1 = vnorm(vcross(galPole, v3(1, 0, 0))), e2 = vcross(galPole, e1);
    int i, j, k, s;
    glDepthMask(GL_FALSE);
    /* faint glow of the Milky Way band */
    glBlendFunc(GL_SRC_ALPHA, GL_ONE);
    for (j = -6; j < 6; j++) {
        glBegin(GL_TRIANGLE_STRIP);
        for (i = 0; i <= 120; i++) {
            double lon = 2.0 * M_PI * i / 120;
            for (k = 1; k >= 0; k--) {
                double b = (j + k) * 0.05;
                Vec3 d = vadd(vscale(vadd(vscale(e1, cos(lon)), vscale(e2, sin(lon))), cos(b)), vscale(galPole, sin(b)));
                double a = 0.11 * exp(-(b / 0.12) * (b / 0.12)) * (0.3 + 1.2 * fbm(vscale(d, 5.0), 4));
                glColor4f(0.75f, 0.80f, 1.0f, (float)a);
                glVertex3d(900.0 * d.x, 900.0 * d.y, 900.0 * d.z);
            }
        }
        glEnd();
    }
    glEnable(GL_POINT_SMOOTH);
    for (s = 0; s < 3; s++) {
        glPointSize(sizes[s]);
        glBegin(GL_POINTS);
        for (i = 0; i < NUM_STARS; i++) {
            float b = starBright[i];
            if (b < lims[s] || b >= lims[s + 1]) continue;
            glColor4f(starCol[i][0] * b, starCol[i][1] * b, starCol[i][2] * b, 1.0f);
            glVertex3fv(starPos[i]);
        }
        glEnd();
    }
    glDisable(GL_POINT_SMOOTH);
    glPointSize(1.0f);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glDepthMask(GL_TRUE);
}

static void drawSunGlow(const World *w)
{
    Vec3 right = v3(mvMat[0], mvMat[4], mvMat[8]);
    Vec3 up = v3(mvMat[1], mvMat[5], mvMat[9]);
    int i, seg = 64;
    double R = SUN_RADIUS * 3.0;
    glDepthMask(GL_FALSE);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE);
    if (texGlow) {
        static const double sz[2] = { 4.5, 12.0 };
        static const float al[2] = { 0.95f, 0.35f };
        glEnable(GL_TEXTURE_2D);
        glBindTexture(GL_TEXTURE_2D, texGlow);
        glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE);
        for (i = 0; i < 2; i++) {
            double q = SUN_RADIUS * sz[i];
            glColor4f(1.0f, 0.82f, 0.55f, al[i]);
            glBegin(GL_QUADS);
            glTexCoord2f(0, 0); glVertexV(vadd(w->sun, vadd(vscale(right, -q), vscale(up, -q))));
            glTexCoord2f(1, 0); glVertexV(vadd(w->sun, vadd(vscale(right, q), vscale(up, -q))));
            glTexCoord2f(1, 1); glVertexV(vadd(w->sun, vadd(vscale(right, q), vscale(up, q))));
            glTexCoord2f(0, 1); glVertexV(vadd(w->sun, vadd(vscale(right, -q), vscale(up, q))));
            glEnd();
        }
        glDisable(GL_TEXTURE_2D);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        glDepthMask(GL_TRUE);
        return;
    }
    glBegin(GL_TRIANGLE_FAN);
    glColor4f(1.0f, 0.8f, 0.4f, 0.55f);
    glVertexV(w->sun);
    for (i = 0; i <= seg; i++) {
        double a = 2.0 * M_PI * i / seg;
        glColor4f(1.0f, 0.5f, 0.1f, 0.0f);
        glVertexV(vadd(w->sun, vadd(vscale(right, R * cos(a)), vscale(up, R * sin(a)))));
    }
    glEnd();
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glDepthMask(GL_TRUE);
}

static void drawOrbits(const World *w)
{
    int i, seg = 360;
    Vec3 asc, desc;
    /* Earth's orbit around the Sun (the ecliptic) */
    glColor4f(0.30f, 0.40f, 0.70f, 0.8f);
    glBegin(GL_LINE_LOOP);
    for (i = 0; i < seg; i++) {
        double a = 2.0 * M_PI * i / seg;
        glVertexV(eclToGL(EARTH_ORBIT * cos(a), EARTH_ORBIT * sin(a), 0));
    }
    glEnd();

    /* Moon's orbit: bright above the ecliptic plane, dim below */
    glLineWidth(1.6f);
    glBegin(GL_LINE_STRIP);
    for (i = 0; i <= seg; i++) {
        double u = 2.0 * M_PI * i / seg;
        Vec3 off = moonOffset(w, u, NULL);
        if (off.y >= 0) glColor4f(0.55f, 0.85f, 1.0f, 0.95f);
        else glColor4f(0.30f, 0.40f, 0.55f, 0.9f);
        glVertexV(vadd(w->earth, off));
    }
    glEnd();
    glLineWidth(1.0f);

    /* line of nodes */
    asc = vadd(w->earth, vscale(moonOffset(w, 0.0, NULL), 1.25));
    desc = vadd(w->earth, vscale(moonOffset(w, M_PI, NULL), 1.25));
    glEnable(GL_LINE_STIPPLE);
    glLineStipple(2, 0xAAAA);
    glColor4f(1.0f, 0.9f, 0.3f, 0.8f);
    glBegin(GL_LINES);
    glVertexV(asc); glVertexV(desc);
    glEnd();
    glDisable(GL_LINE_STIPPLE);
    glPointSize(7.0f);
    glBegin(GL_POINTS);
    glColor3f(0.3f, 1.0f, 0.4f); glVertexV(vadd(w->earth, moonOffset(w, 0.0, NULL)));
    glColor3f(1.0f, 0.4f, 0.3f); glVertexV(vadd(w->earth, moonOffset(w, M_PI, NULL)));
    glEnd();
    glPointSize(1.0f);

    /* Earth's rotation axis */
    {
        double t = EARTH_TILT * DEG2RAD;
        Vec3 ax = v3(-sin(t), cos(t), 0);
        glColor4f(0.8f, 0.8f, 0.8f, 0.7f);
        glBegin(GL_LINES);
        glVertexV(vadd(w->earth, vscale(ax, 1.5 * EARTH_RADIUS)));
        glVertexV(vadd(w->earth, vscale(ax, -1.5 * EARTH_RADIUS)));
        glEnd();
    }
}

static void drawEclipticDisc(const World *w)
{
    int i, seg = 96;
    double R = MOON_ORBIT * 1.45;
    glDepthMask(GL_FALSE);
    glBegin(GL_TRIANGLE_FAN);
    glColor4f(0.35f, 0.40f, 0.85f, 0.08f);
    glVertexV(w->earth);
    for (i = 0; i <= seg; i++) {
        double a = 2.0 * M_PI * i / seg;
        glColor4f(0.35f, 0.40f, 0.85f, 0.03f);
        glVertexV(vadd(w->earth, v3(R * cos(a), 0, R * sin(a))));
    }
    glEnd();
    glDepthMask(GL_TRUE);
}

/* Translucent cone/frustum along 'dir' starting at 'base'. */
static void drawCone(Vec3 base, Vec3 dir, double r0, double r1, double len,
                     float r, float g, float b, float a0, float a1)
{
    Vec3 p1 = vnorm(vcross(dir, v3(0, 1, 0)));
    Vec3 p2 = vcross(dir, p1);
    Vec3 tip = vadd(base, vscale(dir, len));
    int i, seg = 64;
    glBegin(GL_TRIANGLE_STRIP);
    for (i = 0; i <= seg; i++) {
        double ang = 2.0 * M_PI * i / seg;
        Vec3 radial = vadd(vscale(p1, cos(ang)), vscale(p2, sin(ang)));
        glColor4f(r, g, b, a0);
        glVertexV(vadd(base, vscale(radial, r0)));
        glColor4f(r, g, b, a1);
        glVertexV(vadd(tip, vscale(radial, r1)));
    }
    glEnd();
}

static void drawShadowCones(const World *w)
{
    const double Rs = SUN_RADIUS, Re = EARTH_RADIUS, Rm = MOON_RADIUS;
    Vec3 se = vsub(w->earth, w->sun), sm = vsub(w->moon, w->sun);
    double D = vlen(se), Dm = vlen(sm);
    Vec3 d = vscale(se, 1.0 / D), dm = vscale(sm, 1.0 / Dm);
    double Lu = Re * D / (Rs - Re), Lp = MOON_ORBIT * 1.5;
    double Lmu = Rm * Dm / (Rs - Rm), Lmp = vlen(vsub(w->earth, w->moon)) + 2.0;

    glDepthMask(GL_FALSE);
    glDisable(GL_CULL_FACE);
    /* Earth: penumbra (blue) and umbra (red) */
    drawCone(w->earth, d, Re, Re + Lp * (Rs + Re) / D, Lp, 0.55f, 0.65f, 1.0f, 0.07f, 0.01f);
    drawCone(w->earth, d, Re, Re * (1.0 - Lp / Lu), Lp, 1.0f, 0.25f, 0.20f, 0.11f, 0.05f);
    /* Moon: penumbra and umbra */
    drawCone(w->moon, dm, Rm, Rm + Lmp * (Rs + Rm) / Dm, Lmp, 0.55f, 0.65f, 1.0f, 0.08f, 0.02f);
    drawCone(w->moon, dm, Rm, 0.0, Lmu, 1.0f, 0.25f, 0.20f, 0.28f, 0.10f);
    glDepthMask(GL_TRUE);
}

static void drawShadowSpot(const World *w, const EclipseInfo *e)
{
    Vec3 n, p1, p2, c;
    double r;
    int i, seg = 32;
    if (!e->hasShadowPoint) return;
    n = vnorm(vsub(e->shadowPoint, w->earth));
    p1 = vnorm(vcross(n, v3(0, 1, 0)));
    p2 = vcross(n, p1);
    c = vadd(w->earth, vscale(n, EARTH_RADIUS * 1.004));
    r = e->shadowSpotR > 0.035 ? e->shadowSpotR : 0.035;
    glBegin(GL_TRIANGLE_FAN);
    if (e->solar == EC_SOLAR_TOTAL) glColor3f(0.0f, 0.0f, 0.0f); else glColor3f(0.35f, 0.2f, 0.0f);
    glVertexV(c);
    for (i = 0; i <= seg; i++) {
        double a = 2.0 * M_PI * i / seg;
        glVertexV(vadd(c, vadd(vscale(p1, r * cos(a)), vscale(p2, r * sin(a)))));
    }
    glEnd();
}

/* ------------------------------------------------------------------ */
/* Insets: "Sun seen from Earth" and "Moon seen from Earth"            */
/* ------------------------------------------------------------------ */
static void insetFrame(int x0, int y0, int size, const char *title)
{
    strokeRect((float)x0, (float)y0, (float)size, (float)size, 0.6f, 0.7f, 0.9f, 0.9f);
    fillRect((float)x0, (float)y0, (float)size, 18.0f, 0.0f, 0.0f, 0.0f, 0.6f);
    glColor3f(0.9f, 0.95f, 1.0f);
    drawText((float)x0 + 6, (float)y0 + 13, GLUT_BITMAP_HELVETICA_12, title);
}

static void drawSolarInset(const World *w, const EclipseInfo *e, int x0, int y0, int size)
{
    Vec3 P = e->observer;
    Vec3 sdir = vnorm(vsub(w->sun, P)), mvec = vsub(w->moon, P);
    double dS = vlen(vsub(w->sun, P)), dM = vlen(mvec);
    Vec3 mdir = vscale(mvec, 1.0 / dM);
    double a = asin(SUN_RADIUS / dS), b = asin(MOON_RADIUS / dM);
    Vec3 right = vnorm(vcross(sdir, v3(0, 1, 0)));
    Vec3 upv = vcross(right, sdir);
    double cov = e->solarCov, sky = pow(clampd(1.0 - cov, 0, 1), 0.3);
    float cx = x0 + size * 0.5f, cy = y0 + size * 0.5f + 8.0f;
    float sunPx = size * 0.17f;
    double scl = sunPx / a;
    int i;

    fillRect((float)x0, (float)y0, (float)size, (float)size,
             (float)(0.02 + 0.33 * sky), (float)(0.02 + 0.53 * sky), (float)(0.08 + 0.82 * sky), 1.0f);
    glEnable(GL_SCISSOR_TEST);
    glScissor(x0, winH - (y0 + size), size, size);

    if (cov > 0.995) {  /* corona */
        glBegin(GL_TRIANGLE_FAN);
        glColor4f(0.95f, 0.95f, 1.0f, 0.9f);
        glVertex2f(cx, cy);
        for (i = 0; i <= 72; i++) {
            double ang = 2.0 * M_PI * i / 72;
            double rr = sunPx * (2.3 + 0.5 * sin(ang * 5.0) * sin(ang * 3.0));
            glColor4f(0.8f, 0.85f, 1.0f, 0.0f);
            glVertex2f(cx + (float)(rr * cos(ang)), cy + (float)(rr * sin(ang)));
        }
        glEnd();
    }
    /* Sun disc with limb darkening */
    if (texSun) {
        int ri, sj, kk;
        glEnable(GL_TEXTURE_2D);
        glBindTexture(GL_TEXTURE_2D, texSun);
        glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE);
        for (ri = 0; ri < 16; ri++) {
            glBegin(GL_TRIANGLE_STRIP);
            for (sj = 0; sj <= 64; sj++) {
                double ang = 2.0 * M_PI * sj / 64;
                for (kk = 1; kk >= 0; kk--) {
                    double r = (double)(ri + kk) / 16, sx = r * cos(ang), sy = r * sin(ang);
                    double mu = sqrt(clampd(1.0 - r * r, 0, 1)), ld = 0.40 + 0.60 * sqrt(mu);
                    glColor3f((float)clampd(ld * 1.2, 0, 1), (float)clampd(ld * 1.05, 0, 1), (float)clampd(ld * 0.95, 0, 1));
                    glTexCoord2d(0.5 + atan2(sx, mu) / (2.0 * M_PI), 0.5 - asin(sy) / M_PI);
                    glVertex2f(cx + (float)(sx * sunPx), cy - (float)(sy * sunPx));
                }
            }
            glEnd();
        }
        glDisable(GL_TEXTURE_2D);
    } else {
        glBegin(GL_TRIANGLE_FAN);
        glColor3f(1.0f, 1.0f, 0.85f);
        glVertex2f(cx, cy);
        glColor3f(1.0f, 0.62f, 0.15f);
        for (i = 0; i <= 64; i++) {
            double ang = 2.0 * M_PI * i / 64;
            glVertex2f(cx + sunPx * (float)cos(ang), cy + sunPx * (float)sin(ang));
        }
        glEnd();
    }
    /* Moon silhouette */
    if (vdot(mdir, sdir) > 0) {
        float mx = (float)(vdot(mdir, right) * scl), my = (float)(vdot(mdir, upv) * scl);
        glColor3f(0.04f, 0.04f, 0.06f);
        fillCircle2D(cx + mx, cy - my, (float)(b * scl), 64);
    }
    glDisable(GL_SCISSOR_TEST);

    insetFrame(x0, y0, size, "Sun seen from Earth");
    glColor3f(1.0f, 1.0f, 1.0f);
    if (e->solar != EC_NONE || cov > 0.0005)
        drawTextf((float)x0 + 6, (float)(y0 + size - 8), GLUT_BITMAP_HELVETICA_12, "Sun covered: %.1f%%", cov * 100.0);
    else
        drawText((float)x0 + 6, (float)(y0 + size - 8), GLUT_BITMAP_HELVETICA_12, "Moon is not in front of the Sun");
    drawTextf((float)x0 + 6, (float)(y0 + size - 24), GLUT_BITMAP_HELVETICA_10, "Moon/Sun apparent size: %.2f", b / a);
}

static void drawLunarInset(const World *w, int x0, int y0, int size)
{
    Vec3 view = vnorm(vsub(w->moon, w->earth));
    Vec3 right = vnorm(vcross(view, v3(0, 1, 0)));
    Vec3 upv = vcross(right, view);
    float cx = x0 + size * 0.5f, cy = y0 + size * 0.5f + 8.0f;
    float moonPx = size * 0.19f;
    double pxPerUnit = moonPx / MOON_RADIUS;
    int i, j, k, rings = 24, seg = 72;
    double spin = moonSpin(w);
    Vec3 se = vsub(w->earth, w->sun), d, Q, off;
    double D = vlen(se), x, ru, rp;

    fillRect((float)x0, (float)y0, (float)size, (float)size, 0.0f, 0.0f, 0.03f, 1.0f);
    glEnable(GL_SCISSOR_TEST);
    glScissor(x0, winH - (y0 + size), size, size);

    /* Moon disc, shaded with the same lighting as the 3D model (phases + Earth's shadow) */
    if (texMoon) {
        glEnable(GL_TEXTURE_2D);
        glBindTexture(GL_TEXTURE_2D, texMoon);
        glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE);
    }
    for (i = 0; i < rings; i++) {
        glBegin(GL_TRIANGLE_STRIP);
        for (j = 0; j <= seg; j++) {
            double ang = 2.0 * M_PI * j / seg;
            for (k = 1; k >= 0; k--) {
                double r = (double)(i + k) / rings;
                double sx = r * cos(ang), sy = r * sin(ang);
                double sz = sqrt(clampd(1.0 - sx * sx - sy * sy, 0, 1));
                Vec3 n = vadd(vadd(vscale(right, sx), vscale(upv, sy)), vscale(view, -sz));
                Vec3 P = vadd(w->moon, vscale(n, MOON_RADIUS));
                float rgb[3];
                double lon = atan2(-n.z, n.x) - spin;
                moonColor(w, P, n, n, rgb);
                lon = fmod(lon + 3.0 * M_PI, 2.0 * M_PI) - M_PI;
                glColor3fv(rgb);
                glTexCoord2d(0.5 + lon / (2.0 * M_PI), 0.5 - asin(clampd(n.y, -1, 1)) / M_PI);
                glVertex2f(cx + (float)(sx * moonPx), cy - (float)(sy * moonPx));
            }
        }
        glEnd();
    }
    glDisable(GL_TEXTURE_2D);

    /* outlines of Earth's umbra and penumbra at the Moon's distance */
    d = vscale(se, 1.0 / D);
    x = vdot(vsub(w->moon, w->earth), d);
    if (x > 0) {
        float ox, oy;
        Q = vadd(w->earth, vscale(d, x));
        off = vsub(Q, w->moon);
        ox = (float)(vdot(off, right) * pxPerUnit);
        oy = (float)(vdot(off, upv) * pxPerUnit);
        ru = EARTH_RADIUS - x * (SUN_RADIUS - EARTH_RADIUS) / D;
        rp = EARTH_RADIUS + x * (SUN_RADIUS + EARTH_RADIUS) / D;
        glEnable(GL_LINE_STIPPLE);
        glLineStipple(1, 0xCCCC);
        glColor4f(1.0f, 0.35f, 0.3f, 0.9f);
        strokeCircle2D(cx + ox, cy - oy, (float)(ru * pxPerUnit), 128);
        glColor4f(0.6f, 0.7f, 1.0f, 0.8f);
        strokeCircle2D(cx + ox, cy - oy, (float)(rp * pxPerUnit), 128);
        glDisable(GL_LINE_STIPPLE);
    }
    glDisable(GL_SCISSOR_TEST);

    insetFrame(x0, y0, size, "Moon seen from Earth");
    glColor3f(1.0f, 0.5f, 0.45f);
    drawText((float)x0 + 6, (float)(y0 + size - 22), GLUT_BITMAP_HELVETICA_10, "- - Earth's umbra");
    glColor3f(0.65f, 0.75f, 1.0f);
    drawText((float)x0 + 6, (float)(y0 + size - 8), GLUT_BITMAP_HELVETICA_10, "- - Earth's penumbra");
}

/* ------------------------------------------------------------------ */
/* HUD                                                                 */
/* ------------------------------------------------------------------ */
static const char *eclipseName(EclipseType t)
{
    switch (t) {
    case EC_SOLAR_PARTIAL: return "PARTIAL SOLAR ECLIPSE";
    case EC_SOLAR_ANNULAR: return "ANNULAR SOLAR ECLIPSE";
    case EC_SOLAR_TOTAL: return "TOTAL SOLAR ECLIPSE";
    case EC_LUNAR_PENUMBRAL: return "PENUMBRAL LUNAR ECLIPSE";
    case EC_LUNAR_PARTIAL: return "PARTIAL LUNAR ECLIPSE";
    case EC_LUNAR_TOTAL: return "TOTAL LUNAR ECLIPSE";
    default: return "No eclipse right now";
    }
}

static void lessonText(const World *w, const EclipseInfo *e, const char **l1, const char **l2)
{
    EclipseType t = e->solar != EC_NONE ? e->solar : e->lunar;
    double el = moonElongation(w);
    *l1 = ""; *l2 = "";
    switch (t) {
    case EC_SOLAR_PARTIAL:
        *l1 = "Only the Moon's PENUMBRA (light outer shadow) touches the Earth.";
        *l2 = "People inside it see the Moon cover PART of the Sun."; break;
    case EC_SOLAR_TOTAL:
        *l1 = "The Moon's UMBRA (dark inner shadow) reaches the Earth. Inside the small black spot the";
        *l2 = "Sun is completely hidden, the sky turns dark and the Sun's corona becomes visible."; break;
    case EC_SOLAR_ANNULAR:
        *l1 = "The Moon is far away (near APOGEE): it looks smaller than the Sun and its umbra ends";
        *l2 = "before reaching Earth. A bright 'ring of fire' stays around the Moon."; break;
    case EC_LUNAR_PENUMBRAL:
        *l1 = "The Moon is in Earth's PENUMBRA: the Sun is only partly hidden as seen from the Moon,";
        *l2 = "so the Moon dims slightly. This is hard to notice by eye."; break;
    case EC_LUNAR_PARTIAL:
        *l1 = "Part of the Moon has entered Earth's dark UMBRA. A dark 'bite' is visible on the Moon.";
        *l2 = "The edge of the shadow is curved: Greeks used this to show that the Earth is round."; break;
    case EC_LUNAR_TOTAL:
        *l1 = "The whole Moon is inside Earth's UMBRA. It glows RED ('Blood Moon') because Earth's";
        *l2 = "atmosphere bends red sunlight into the shadow (like sunsets all around the Earth)."; break;
    default:
        if (el < 15 || el > 345) {
            *l1 = "NEW MOON: Sun, Moon and Earth are lined up in longitude, but the Moon passes above";
            *l2 = "or below the Sun because its orbit is tilted - so there is no solar eclipse this time.";
        } else if (el > 165 && el < 195) {
            *l1 = "FULL MOON: the Earth is between the Sun and the Moon, but the Moon passes above or";
            *l2 = "below Earth's shadow because its orbit is tilted - so there is no lunar eclipse this time.";
        } else {
            *l1 = "Solar eclipses need a NEW Moon, lunar eclipses need a FULL Moon - and in both cases";
            *l2 = "the Moon must be close to a NODE (where its orbit crosses the ecliptic plane).";
        }
    }
}

static void drawLabel3D(Vec3 p, const char *txt, float r, float g, float b)
{
    GLdouble sx, sy, sz;
    if (gluProject(p.x, p.y, p.z, mvMat, prMat, viewportArr, &sx, &sy, &sz) == GL_TRUE && sz < 1.0) {
        glColor3f(r, g, b);
        drawText((float)sx + 8, (float)(winH - sy) - 8, GLUT_BITMAP_HELVETICA_12, txt);
    }
}

static void drawHUD(const World *w, const EclipseInfo *e)
{
    double el = moonElongation(w), lit = (1.0 - cos(el * DEG2RAD)) * 0.5;
    double sunLon = w->earthAngle + M_PI, delta, lim, height;
    EclipseType t = e->solar != EC_NONE ? e->solar : e->lunar;
    const char *l1, *l2;
    int y, inset, ix;
    char buf[200];

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0, winW, winH, 0, -1, 1);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glDisable(GL_DEPTH_TEST);

    if (showLabels) {
        drawLabel3D(vadd(w->sun, v3(0, SUN_RADIUS, 0)), "Sun", 1.0f, 0.85f, 0.4f);
        drawLabel3D(vadd(w->earth, v3(0, EARTH_RADIUS, 0)), "Earth", 0.6f, 0.8f, 1.0f);
        drawLabel3D(vadd(w->moon, v3(0, MOON_RADIUS, 0)), "Moon", 0.9f, 0.9f, 0.9f);
        if (showOrbits && w->incl > 0) {
            drawLabel3D(vadd(w->earth, moonOffset(w, 0.0, NULL)), "Ascending node", 0.4f, 1.0f, 0.5f);
            drawLabel3D(vadd(w->earth, moonOffset(w, M_PI, NULL)), "Descending node", 1.0f, 0.5f, 0.4f);
        }
    }

    /* ---- information panel (top-left) ---- */
    fillRect(8, 8, 430, 186, 0.0f, 0.0f, 0.0f, 0.55f);
    glColor3f(1.0f, 0.85f, 0.4f);
    drawText(16, 30, GLUT_BITMAP_HELVETICA_18, "ECLIPSE SIMULATOR  -  Sun, Earth & Moon");
    glColor3f(0.9f, 0.9f, 0.9f);
    y = 52;
    drawTextf(16, (float)y, GLUT_BITMAP_HELVETICA_12, "Time: day %.2f  (year %d, day %d)", w->t,
              (int)floor(w->t / YEAR_DAYS) + 1, (int)fmod(w->t < 0 ? 0 : w->t, YEAR_DAYS) + 1); y += 16;
    if (paused) snprintf(buf, sizeof(buf), "Speed: %.3g days/s   [PAUSED]", speed);
    else if (autoSlow && t != EC_NONE && speed > SLOW_SPEED)
        snprintf(buf, sizeof(buf), "Speed: %.3g days/s  (slowed during eclipse)", SLOW_SPEED);
    else snprintf(buf, sizeof(buf), "Speed: %.3g days/s", speed);
    drawText(16, (float)y, GLUT_BITMAP_HELVETICA_12, buf); y += 16;
    drawTextf(16, (float)y, GLUT_BITMAP_HELVETICA_12, "Moon phase: %s  (%.0f%% lit, elongation %.0f deg)",
              phaseName(el), lit * 100.0, el); y += 16;
    drawTextf(16, (float)y, GLUT_BITMAP_HELVETICA_12, "Moon orbit tilt: %s", inclNames[inclIdx]); y += 16;
    drawTextf(16, (float)y, GLUT_BITMAP_HELVETICA_12, "Moon distance: %.2f  (%s orbit%s)", w->moonDist,
              eccOn ? "elliptical" : "circular",
              !eccOn ? "" : (w->moonDist < MOON_ORBIT * (1 - MOON_ECC * 0.7) ? ", near PERIGEE"
                           : (w->moonDist > MOON_ORBIT * (1 + MOON_ECC * 0.7) ? ", near APOGEE" : ""))); y += 16;
    height = w->moon.y - w->earth.y;
    drawTextf(16, (float)y, GLUT_BITMAP_HELVETICA_12, "Moon is %.2f units %s the ecliptic plane",
              fabs(height), height >= 0 ? "above" : "below"); y += 16;
    delta = fmod(fabs(sunLon - w->nodeLon), M_PI);
    if (delta > M_PI / 2) delta = M_PI - delta;
    lim = EARTH_RADIUS + MOON_ORBIT * (SUN_RADIUS + EARTH_RADIUS) / EARTH_ORBIT + MOON_RADIUS;
    lim = w->incl > 1e-6 ? asin(clampd(lim / (MOON_ORBIT * sin(w->incl)), 0, 1)) : M_PI / 2;
    if (delta <= lim) glColor3f(1.0f, 0.85f, 0.3f);
    drawTextf(16, (float)y, GLUT_BITMAP_HELVETICA_12, "Sun is %.0f deg from the line of nodes%s",
              delta * RAD2DEG, delta <= lim ? "  ->  ECLIPSE SEASON" : ""); y += 16;
    glColor3f(0.7f, 0.7f, 0.7f);
    drawTextf(16, (float)y, GLUT_BITMAP_HELVETICA_12, "Camera: %s", camNames[camMode]); y += 16;

    /* ---- eclipse status ---- */
    fillRect(8, 200, 430, 54, 0.0f, 0.0f, 0.0f, 0.55f);
    if (isSolar(t)) glColor3f(1.0f, 0.65f, 0.15f);
    else if (t != EC_NONE) glColor3f(1.0f, 0.3f, 0.25f);
    else glColor3f(0.6f, 0.6f, 0.6f);
    drawText(16, 224, GLUT_BITMAP_HELVETICA_18, eclipseName(t));
    glColor3f(0.9f, 0.9f, 0.9f);
    if (isSolar(t)) drawTextf(16, 244, GLUT_BITMAP_HELVETICA_12, "New Moon: Sun - Moon - Earth in line.  Max. Sun covered: %.1f%%", e->solarCov * 100.0);
    else if (t != EC_NONE) drawTextf(16, 244, GLUT_BITMAP_HELVETICA_12, "Full Moon: Sun - Earth - Moon in line.  Umbral magnitude: %.2f", e->lunarMag);
    else drawText(16, 244, GLUT_BITMAP_HELVETICA_12, "Press 2 for the next solar eclipse, 3 for the next lunar eclipse");

    /* ---- lesson text (bottom) ---- */
    lessonText(w, e, &l1, &l2);
    {
        int lw = textWidth(GLUT_BITMAP_HELVETICA_12, l1), lw2 = textWidth(GLUT_BITMAP_HELVETICA_12, l2);
        int bw = (lw > lw2 ? lw : lw2) + 24;
        int bx = (winW - bw) / 2, by = winH - 54;
        fillRect((float)bx, (float)by, (float)bw, 44, 0.05f, 0.08f, 0.18f, 0.75f);
        strokeRect((float)bx, (float)by, (float)bw, 44, 0.4f, 0.5f, 0.8f, 0.8f);
        glColor3f(1.0f, 1.0f, 1.0f);
        drawText((float)bx + 12, (float)by + 18, GLUT_BITMAP_HELVETICA_12, l1);
        drawText((float)bx + 12, (float)by + 35, GLUT_BITMAP_HELVETICA_12, l2);
    }

    /* ---- insets (right) ---- */
    inset = winH / 3 - 20;
    if (inset > 240) inset = 240;
    if (inset < 140) inset = 140;
    ix = winW - inset - 10;
    drawSolarInset(w, e, ix, 10, inset);
    drawLunarInset(w, ix, 20 + inset, inset);

    /* ---- legend ---- */
    if (showShadows) {
        int ly = 30 + 2 * inset;
        fillRect((float)ix, (float)ly, (float)inset, 46, 0.0f, 0.0f, 0.0f, 0.55f);
        fillRect((float)ix + 8, (float)ly + 8, 14, 12, 1.0f, 0.25f, 0.2f, 0.8f);
        glColor3f(0.95f, 0.95f, 0.95f);
        drawText((float)ix + 28, (float)ly + 18, GLUT_BITMAP_HELVETICA_12, "Umbra (full shadow)");
        fillRect((float)ix + 8, (float)ly + 26, 14, 12, 0.55f, 0.65f, 1.0f, 0.6f);
        glColor3f(0.95f, 0.95f, 0.95f);
        drawText((float)ix + 28, (float)ly + 36, GLUT_BITMAP_HELVETICA_12, "Penumbra (partial shadow)");
    }

    /* ---- help (bottom-left) ---- */
    if (showHelp) {
        static const char *help[] = {
            "CONTROLS",
            "Space   pause / resume",
            "+ / -   faster / slower",
            ", / .   step back / forward (6 h)",
            "1       free run (normal speed)",
            "2 / 3   jump to next SOLAR / LUNAR eclipse",
            "i       change Moon orbit tilt",
            "e       circular / elliptical Moon orbit",
            "c       change camera",
            "Mouse drag / arrows   rotate view",
            "Wheel / z x           zoom",
            "o  orbits   s  shadows   l  labels",
            "a  auto-slow during eclipses",
            "f  eclipse facts    h  hide help",
            "r  reset    q / Esc  quit"
        };
        int n = (int)(sizeof(help) / sizeof(help[0])), i;
        int hy = winH - 66 - n * 15;
        fillRect(8, (float)hy - 16, 290, (float)n * 15 + 10, 0.0f, 0.0f, 0.0f, 0.55f);
        for (i = 0; i < n; i++) {
            if (i == 0) glColor3f(1.0f, 0.85f, 0.4f); else glColor3f(0.85f, 0.85f, 0.85f);
            drawText(16, (float)(hy + i * 15), GLUT_BITMAP_HELVETICA_12, help[i]);
        }
    } else {
        glColor3f(0.7f, 0.7f, 0.7f);
        drawText(12, (float)winH - 66, GLUT_BITMAP_HELVETICA_12, "h: show controls   f: eclipse facts");
    }

    /* ---- facts overlay ---- */
    if (showFacts) {
        static const char *facts[] = {
            "HOW ECLIPSES WORK",
            "",
            "* A SOLAR eclipse happens at NEW MOON: the Moon passes between the Sun and the Earth and",
            "   its shadow falls on part of the Earth.",
            "* A LUNAR eclipse happens at FULL MOON: the Earth passes between the Sun and the Moon and",
            "   the Moon moves into the Earth's shadow. Everyone on the night side can see it.",
            "* UMBRA = the dark inner part of a shadow (the Sun is completely blocked).",
            "* PENUMBRA = the lighter outer part of a shadow (the Sun is only partly blocked).",
            "* TOTAL solar eclipse: you are inside the Moon's umbra. It covers only a narrow path on Earth.",
            "* ANNULAR solar eclipse: the Moon is near APOGEE (farthest point) and looks smaller than the",
            "   Sun, so a 'ring of fire' remains.",
            "* In a total lunar eclipse the Moon turns RED: Earth's atmosphere bends (refracts) red light",
            "   into the umbra, while blue light is scattered away.",
            "* Eclipses do NOT happen every month because the Moon's orbit is tilted by about 5.1 deg",
            "   to the ecliptic. The Moon crosses the ecliptic at two NODES; an eclipse can only happen",
            "   when a New or Full Moon occurs close to a node ('eclipse season', about twice a year).",
            "* The Sun is about 400 times wider than the Moon but also about 400 times farther away,",
            "   so both look almost the same size in our sky - that is why total eclipses are possible.",
            "* NEVER look at the Sun directly, not even during a partial eclipse. Use eclipse glasses.",
            "",
            "NOTE: sizes and distances in this simulator are NOT to scale.",
            "Press f to close."
        };
        int n = (int)(sizeof(facts) / sizeof(facts[0])), i, fw = 640, fh = n * 17 + 30;
        int fx = (winW - fw) / 2, fy = (winH - fh) / 2;
        fillRect((float)fx, (float)fy, (float)fw, (float)fh, 0.02f, 0.04f, 0.12f, 0.92f);
        strokeRect((float)fx, (float)fy, (float)fw, (float)fh, 0.5f, 0.6f, 1.0f, 1.0f);
        for (i = 0; i < n; i++) {
            if (i == 0) glColor3f(1.0f, 0.85f, 0.4f); else glColor3f(0.92f, 0.92f, 0.95f);
            drawText((float)fx + 18, (float)(fy + 26 + i * 17), i == 0 ? GLUT_BITMAP_HELVETICA_18 : GLUT_BITMAP_HELVETICA_12, facts[i]);
        }
    }

    /* ---- transient status message ---- */
    if (statusMsg[0] && glutGet(GLUT_ELAPSED_TIME) < statusMsgUntil) {
        int sw = textWidth(GLUT_BITMAP_HELVETICA_18, statusMsg) + 30;
        int sx = (winW - sw) / 2;
        fillRect((float)sx, 12, (float)sw, 32, 0.1f, 0.1f, 0.25f, 0.85f);
        glColor3f(1.0f, 1.0f, 0.6f);
        drawText((float)sx + 15, 34, GLUT_BITMAP_HELVETICA_18, statusMsg);
    }

    glEnable(GL_DEPTH_TEST);
}

/* ------------------------------------------------------------------ */
/* GLUT callbacks                                                      */
/* ------------------------------------------------------------------ */
static void saveScreenshot(const char *file)
{
    unsigned char *px = (unsigned char *)malloc((size_t)winW * winH * 3);
    FILE *fp;
    int y;
    if (!px) return;
    glPixelStorei(GL_PACK_ALIGNMENT, 1);
    glReadBuffer(GL_BACK);
    glReadPixels(0, 0, winW, winH, GL_RGB, GL_UNSIGNED_BYTE, px);
    fp = fopen(file, "wb");
    if (fp) {
        fprintf(fp, "P6\n%d %d\n255\n", winW, winH);
        for (y = winH - 1; y >= 0; y--) fwrite(px + (size_t)y * winW * 3, 1, (size_t)winW * 3, fp);
        fclose(fp);
    }
    free(px);
}

static void display(void)
{
    World *w = &curWorld;
    EclipseInfo *e = &curInfo;
    Vec3 target;
    double yaw, pitch;

    computeWorld(simTime, w);
    classify(w, e);

    glClearColor(0.0f, 0.0f, 0.02f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(45.0, (double)winW / (winH > 0 ? winH : 1), 0.05, 3000.0);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    target = camMode == 1 ? vscale(vadd(w->sun, w->earth), 0.5) : w->earth;
    yaw = camYaw * DEG2RAD;
    if (camMode == 1) yaw += w->earthAngle + M_PI / 2;   /* look at the Sun from behind the Earth */
    if (camMode == 2) yaw += w->earthAngle;              /* perpendicular to the Sun-Earth line */
    pitch = camPitch * DEG2RAD;
    camPos = vadd(target, vscale(v3(cos(pitch) * sin(yaw), sin(pitch), cos(pitch) * cos(yaw)), camDist));
    gluLookAt(camPos.x, camPos.y, camPos.z, target.x, target.y, target.z, 0, 1, 0);
    glGetDoublev(GL_MODELVIEW_MATRIX, mvMat);
    glGetDoublev(GL_PROJECTION_MATRIX, prMat);
    glGetIntegerv(GL_VIEWPORT, viewportArr);

    /* stars stay fixed relative to the camera position */
    glPushMatrix();
    glTranslated(camPos.x, camPos.y, camPos.z);
    drawStars();
    glPopMatrix();

    drawSphereColored(w, w->sun, SUN_RADIUS, 64, 32, 2.0 * M_PI * w->t / 25.4, 0, sunColor, texSun);
    /* translucent helpers first, then the bodies on top so shadows on their surfaces stay visible */
    if (showOrbits) drawEclipticDisc(w);
    if (showShadows) drawShadowCones(w);
    drawEarth(w);
    drawSphereColored(w, w->moon, MOON_RADIUS, 96, 48, moonSpin(w), 0, moonColor, texMoon);
    drawShadowSpot(w, e);

    if (showOrbits) drawOrbits(w);
    drawSunGlow(w);

    drawHUD(w, e);

    if (shotFile && ++shotFrames >= 2) {
        saveScreenshot(shotFile);
        exit(0);
    }
    glutSwapBuffers();
}

static void reshape(int w, int h)
{
    winW = w; winH = h > 0 ? h : 1;
    glViewport(0, 0, winW, winH);
}

static void idle(void)
{
    int now = glutGet(GLUT_ELAPSED_TIME);
    double dt = (now - lastMs) / 1000.0;
    lastMs = now;
    if (dt > 0.1) dt = 0.1;
    if (!paused) {
        double sp = speed;
        int inEclipse = curInfo.solar != EC_NONE || curInfo.lunar != EC_NONE;
        if (autoSlow && inEclipse && sp > SLOW_SPEED) sp = SLOW_SPEED;
        simTime += sp * dt;
    }
    glutPostRedisplay();
}

static void keyboard(unsigned char key, int x, int y)
{
    char buf[160];
    (void)x; (void)y;
    switch (key) {
    case 27: case 'q': case 'Q': exit(0);
    case ' ': paused = !paused; break;
    case '+': case '=': speed *= 2.0; if (speed > 64.0) speed = 64.0; break;
    case '-': case '_': speed *= 0.5; if (speed < 0.01) speed = 0.01; break;
    case ',': case '<': simTime -= 0.25; break;
    case '.': case '>': simTime += 0.25; break;
    case '1': speed = 1.0; paused = 0; setStatus("Free run: 1 day per second"); break;
    case '2': jumpToNext(1); break;
    case '3': jumpToNext(0); break;
    case 'i': case 'I':
        inclIdx = (inclIdx + 1) % 4;
        snprintf(buf, sizeof(buf), "Moon orbit tilt: %s", inclNames[inclIdx]);
        setStatus(buf); break;
    case 'e': case 'E':
        eccOn = !eccOn;
        setStatus(eccOn ? "Elliptical Moon orbit: total AND annular eclipses"
                        : "Circular Moon orbit: Moon and Sun look almost the same size");
        break;
    case 'c': case 'C': setCamera((camMode + 1) % 3); setStatus(camNames[camMode]); break;
    case 'o': case 'O': showOrbits = !showOrbits; break;
    case 's': case 'S': showShadows = !showShadows; break;
    case 'l': case 'L': showLabels = !showLabels; break;
    case 'a': case 'A': autoSlow = !autoSlow; setStatus(autoSlow ? "Auto-slow ON" : "Auto-slow OFF"); break;
    case 'h': case 'H': showHelp = !showHelp; break;
    case 'f': case 'F': showFacts = !showFacts; break;
    case 'r': case 'R': resetSim(); break;
    case 'z': case 'Z': camDist *= 0.9; break;
    case 'x': case 'X': camDist /= 0.9; break;
    default: break;
    }
    camDist = clampd(camDist, 2.5, 600.0);
}

static void special(int key, int x, int y)
{
    (void)x; (void)y;
    switch (key) {
    case GLUT_KEY_LEFT: camYaw -= 5; break;
    case GLUT_KEY_RIGHT: camYaw += 5; break;
    case GLUT_KEY_UP: camPitch += 5; break;
    case GLUT_KEY_DOWN: camPitch -= 5; break;
    case GLUT_KEY_PAGE_UP: camDist *= 0.9; break;
    case GLUT_KEY_PAGE_DOWN: camDist /= 0.9; break;
    default: break;
    }
    camPitch = clampd(camPitch, -89, 89);
    camDist = clampd(camDist, 2.5, 600.0);
}

static void mouse(int button, int state, int x, int y)
{
    if (button == 3 && state == GLUT_DOWN) camDist *= 0.9;          /* wheel up */
    else if (button == 4 && state == GLUT_DOWN) camDist /= 0.9;     /* wheel down */
    else if (state == GLUT_DOWN) { mouseBtn = button; mouseX = x; mouseY = y; }
    else mouseBtn = -1;
    camDist = clampd(camDist, 2.5, 600.0);
}

static void motion(int x, int y)
{
    int dx = x - mouseX, dy = y - mouseY;
    mouseX = x; mouseY = y;
    if (mouseBtn == GLUT_LEFT_BUTTON) {
        camYaw -= dx * 0.4;
        camPitch = clampd(camPitch + dy * 0.4, -89, 89);
    } else if (mouseBtn == GLUT_RIGHT_BUTTON) {
        camDist = clampd(camDist * (1.0 + dy * 0.01), 2.5, 600.0);
    }
}

static void usage(const char *prog)
{
    printf("Usage: %s [options]\n"
           "  --solar          start at the next solar eclipse\n"
           "  --lunar          start at the next lunar eclipse\n"
           "  --time DAYS      start time in days (applied before --solar/--lunar)\n"
           "  --advance DAYS   advance the clock by DAYS after the jump\n"
           "  --tilt N         Moon orbit tilt option 0..3\n"
           "  --circular       circular Moon orbit\n"
           "  --camera N       camera mode 0..2\n"
           "  --paused         start paused\n"
           "  --yaw/--pitch/--dist X   camera angles (deg) and distance\n"
           "  --facts          show the facts panel\n"
           "  --list YEARS     print a table of upcoming eclipses and exit\n"
           "  --shot FILE.ppm  render one frame to a PPM image and exit\n", prog);
}

int main(int argc, char **argv)
{
    int i, jump = -1, cam = -1, startPaused = 0;
    double advance = 0.0, listYears = 0.0, yawOpt = -1000, pitchOpt = -1000, distOpt = 0;

    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA | GLUT_DEPTH | GLUT_MULTISAMPLE);
    glutInitWindowSize(winW, winH);
    glutCreateWindow("Eclipse Simulator - Solar & Lunar Eclipses (OpenGL)");

    setCamera(2);
    for (i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--solar")) jump = 1;
        else if (!strcmp(argv[i], "--lunar")) jump = 0;
        else if (!strcmp(argv[i], "--time") && i + 1 < argc) simTime = atof(argv[++i]);
        else if (!strcmp(argv[i], "--advance") && i + 1 < argc) advance = atof(argv[++i]);
        else if (!strcmp(argv[i], "--tilt") && i + 1 < argc) inclIdx = (int)clampd(atoi(argv[++i]), 0, 3);
        else if (!strcmp(argv[i], "--circular")) eccOn = 0;
        else if (!strcmp(argv[i], "--camera") && i + 1 < argc) cam = (int)clampd(atoi(argv[++i]), 0, 2);
        else if (!strcmp(argv[i], "--paused")) startPaused = 1;
        else if (!strcmp(argv[i], "--yaw") && i + 1 < argc) yawOpt = atof(argv[++i]);
        else if (!strcmp(argv[i], "--pitch") && i + 1 < argc) pitchOpt = atof(argv[++i]);
        else if (!strcmp(argv[i], "--dist") && i + 1 < argc) distOpt = atof(argv[++i]);
        else if (!strcmp(argv[i], "--facts")) showFacts = 1;
        else if (!strcmp(argv[i], "--shot") && i + 1 < argc) shotFile = argv[++i];
        else if (!strcmp(argv[i], "--list") && i + 1 < argc) listYears = atof(argv[++i]);
        else { usage(argv[0]); return 1; }
    }
    if (listYears > 0) { listEclipses(listYears); return 0; }
    if (jump >= 0) jumpToNext(jump);
    if (startPaused) paused = 1;
    simTime += advance;
    if (cam >= 0) setCamera(cam);
    if (yawOpt > -999) camYaw = yawOpt;
    if (pitchOpt > -999) camPitch = pitchOpt;
    if (distOpt > 0) camDist = distOpt;
    if (shotFile) statusMsg[0] = 0;

    initStars();
    initTextures(argv[0]);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glEnable(GL_LINE_SMOOTH);
    glShadeModel(GL_SMOOTH);
#ifdef GL_MULTISAMPLE
    glEnable(GL_MULTISAMPLE);
#endif

    glutDisplayFunc(display);
    glutReshapeFunc(reshape);
    glutIdleFunc(idle);
    glutKeyboardFunc(keyboard);
    glutSpecialFunc(special);
    glutMouseFunc(mouse);
    glutMotionFunc(motion);

    lastMs = glutGet(GLUT_ELAPSED_TIME);
    glutMainLoop();
    return 0;
}
