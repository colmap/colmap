#ifndef LITE_WINDOW_H
#define LITE_WINDOW_H

//#define WINDOW_PREFER_GLUT

#if defined(WINDOW_PREFER_GLUT)

#ifdef __APPLE__
	#include "GLUT/glut.h"
#else
	#include "GL/glut.h"
#endif
//for apple, use GLUT to create the window..
class LiteWindow
{
    int glut_id;
public:
    LiteWindow()            {  glut_id = 0;         }
    int IsValid()           {  return glut_id > 0; }
    virtual ~LiteWindow()   {  if(glut_id > 0) glutDestroyWindow(glut_id);  }
    void MakeCurrent()      {  glutSetWindow(glut_id);    }
    void Create(int x = -1, int y = -1, const char* display = NULL)
    {
	    static int _glut_init_called = 0;
        if(glut_id != 0) return;

	    //see if there is an existing window
	    if(_glut_init_called) glut_id = glutGetWindow();

	    //create one if no glut window exists
	    if(glut_id != 0) return;

	    if(_glut_init_called == 0)
	    {
		    int argc = 1;
		    char * argv[4] = { "-iconic", 0 , 0, 0};
            if(display)
            {
                argc = 3;
                argv[1] = "-display";
                argv[2] = (char*) display;
            }
		    glutInit(&argc, argv);
		    glutInitDisplayMode(GLUT_RGBA);
		    _glut_init_called = 1;
	    }
	    if(x != -1) glutInitWindowPosition(x, y);
        if(display || x != -1) std::cout << "Using display ["
            << (display? display : "\0" )<< "] at (" << x << "," << y << ")\n";
	    glut_id = glutCreateWindow("SIFT_GPU_GLUT");
	    glutHideWindow();
    }
};
#elif defined( _WIN32)

#ifndef _INC_WINDOWS
#ifndef WIN32_LEAN_AND_MEAN
	#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#endif

class LiteWindow
{
    HWND hWnd;
    HGLRC hContext;
    HDC hdc;
public:
    LiteWindow()
    {
        hWnd = NULL;
        hContext = NULL;
        hdc = NULL;
    }
    virtual ~LiteWindow()
    {
        if(hContext)wglDeleteContext(hContext);
        if(hdc)ReleaseDC(hWnd, hdc);
        if(hWnd)DestroyWindow(hWnd);
    }
    int IsValid()
    {
        return hContext != NULL;
    }

    //display is ignored under Win32
    void Create(int x = -1, int y = -1, const char* display = NULL)
    {
        if(hContext) return;
        WNDCLASSEX wcex = { sizeof(WNDCLASSEX),  CS_HREDRAW | CS_VREDRAW,
                            (WNDPROC)DefWindowProc,  0, 4, 0, 0, 0, 0, 0,
                            ("SIFT_GPU_LITE"),    0};
        RegisterClassEx(&wcex);
        hWnd = CreateWindow("SIFT_GPU_LITE", "SIFT_GPU", 0,
                            CW_USEDEFAULT, CW_USEDEFAULT,
                            100, 100, NULL, NULL, 0, 0);

        //move the window so that it can be on the second monitor
        if(x !=-1)
        {
            MoveWindow(hWnd, x, y, 100, 100, 0);
            std::cout << "CreateWindow at (" << x << "," << y<<")\n";
        }

        ///////////////////////////////////////////////////
        PIXELFORMATDESCRIPTOR pfd =
        {
            sizeof(PIXELFORMATDESCRIPTOR), 1,
            PFD_DRAW_TO_WINDOW | PFD_SUPPORT_OPENGL ,
            PFD_TYPE_RGBA,16,0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 16, 0, 0, 0, 0, 0, 0, 0
        };
        hdc=GetDC(hWnd);
        ////////////////////////////////////
        int pixelformat = ChoosePixelFormat(hdc, &pfd);
        DescribePixelFormat(hdc, pixelformat, sizeof(pfd), &pfd);
        SetPixelFormat(hdc, pixelformat, &pfd);
        hContext = wglCreateContext(hdc);

    }
    void MakeCurrent()
    {
        wglMakeCurrent(hdc, hContext);
    }
};

#else

#include <unistd.h>
#include <X11/Xlib.h>
#include <GL/glx.h>

class LiteWindow
{
    Display*     xDisplay;
    XVisualInfo* xVisual;
    Window       xWin;
    GLXContext   xContext;
    Colormap     xColormap;
public:
    LiteWindow()
    {
        xDisplay = NULL;
        xVisual = NULL;
        xWin = 0;
        xColormap = 0;
        xContext = NULL;
    }
    int IsValid ()
    {
        return xContext != NULL  && glXIsDirect(xDisplay, xContext);
    }
    virtual ~LiteWindow()
    {
        if(xWin) XDestroyWindow(xDisplay, xWin);
        if(xContext) glXDestroyContext(xDisplay, xContext);
        if(xColormap) XFreeColormap(xDisplay, xColormap);
        if(xDisplay) XCloseDisplay(xDisplay);
    }
    void Create(int x = 0, int y = 0, const char * display = NULL)
    {
        if(xDisplay) return;
        if(display) std::cout << "Using display ["<<display<<"]\n";

        xDisplay = XOpenDisplay(display && display[0] ? display : NULL);
        if(xDisplay == NULL) return;
        int attrib[] =  {GLX_RGBA, GLX_RED_SIZE, 1,
                         GLX_GREEN_SIZE, 1, GLX_BLUE_SIZE, 1,  0 };
        xVisual = glXChooseVisual(xDisplay, DefaultScreen(xDisplay), attrib);
        if(xVisual == NULL) return;
        xColormap = XCreateColormap(
            xDisplay, RootWindow(xDisplay, xVisual->screen),
            xVisual->visual, AllocNone);

        XSetWindowAttributes wa;
        wa.event_mask       = 0;
        wa.border_pixel     = 0;
        wa.colormap = xColormap;

        xWin = XCreateWindow( xDisplay, RootWindow(xDisplay, xVisual->screen) ,
                              x, y, 100, 100, 0, xVisual->depth,
                              InputOutput, xVisual->visual,
                              CWBorderPixel |CWColormap | CWEventMask, &wa);

        xContext = glXCreateContext(xDisplay, xVisual,  0, GL_TRUE);
    }
    void MakeCurrent()
    {
        if(xContext) glXMakeCurrent(xDisplay, xWin, xContext);
    }
};

#endif


#endif

