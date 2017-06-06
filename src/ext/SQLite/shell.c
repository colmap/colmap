/*
** 2001 September 15
**
** The author disclaims copyright to this source code.  In place of
** a legal notice, here is a blessing:
**
**    May you do good and not evil.
**    May you find forgiveness for yourself and forgive others.
**    May you share freely, never taking more than you give.
**
*************************************************************************
** This file contains code to implement the "sqlite" command line
** utility for accessing SQLite databases.
*/
#if (defined(_WIN32) || defined(WIN32)) && !defined(_CRT_SECURE_NO_WARNINGS)
/* This needs to come before any includes for MSVC compiler */
#define _CRT_SECURE_NO_WARNINGS
#endif

/*
** If requested, include the SQLite compiler options file for MSVC.
*/
#if defined(INCLUDE_MSVC_H)
#include "msvc.h"
#endif

/*
** No support for loadable extensions in VxWorks.
*/
#if (defined(__RTP__) || defined(_WRS_KERNEL)) && !SQLITE_OMIT_LOAD_EXTENSION
# define SQLITE_OMIT_LOAD_EXTENSION 1
#endif

/*
** Enable large-file support for fopen() and friends on unix.
*/
#ifndef SQLITE_DISABLE_LFS
# define _LARGE_FILE       1
# ifndef _FILE_OFFSET_BITS
#   define _FILE_OFFSET_BITS 64
# endif
# define _LARGEFILE_SOURCE 1
#endif

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <assert.h>
#include "sqlite3.h"
#if SQLITE_USER_AUTHENTICATION
# include "sqlite3userauth.h"
#endif
#include <ctype.h>
#include <stdarg.h>

#if !defined(_WIN32) && !defined(WIN32)
# include <signal.h>
# if !defined(__RTP__) && !defined(_WRS_KERNEL)
#  include <pwd.h>
# endif
# include <unistd.h>
# include <sys/types.h>
#endif

#if HAVE_READLINE
# include <readline/readline.h>
# include <readline/history.h>
#endif

#if HAVE_EDITLINE
# include <editline/readline.h>
#endif

#if HAVE_EDITLINE || HAVE_READLINE

# define shell_add_history(X) add_history(X)
# define shell_read_history(X) read_history(X)
# define shell_write_history(X) write_history(X)
# define shell_stifle_history(X) stifle_history(X)
# define shell_readline(X) readline(X)

#elif HAVE_LINENOISE

# include "linenoise.h"
# define shell_add_history(X) linenoiseHistoryAdd(X)
# define shell_read_history(X) linenoiseHistoryLoad(X)
# define shell_write_history(X) linenoiseHistorySave(X)
# define shell_stifle_history(X) linenoiseHistorySetMaxLen(X)
# define shell_readline(X) linenoise(X)

#else

# define shell_read_history(X)
# define shell_write_history(X)
# define shell_stifle_history(X)

# define SHELL_USE_LOCAL_GETLINE 1
#endif


#if defined(_WIN32) || defined(WIN32)
# include <io.h>
# include <fcntl.h>
# define isatty(h) _isatty(h)
# ifndef access
#  define access(f,m) _access((f),(m))
# endif
# undef popen
# define popen _popen
# undef pclose
# define pclose _pclose
#else
 /* Make sure isatty() has a prototype. */
 extern int isatty(int);

# if !defined(__RTP__) && !defined(_WRS_KERNEL)
  /* popen and pclose are not C89 functions and so are
  ** sometimes omitted from the <stdio.h> header */
   extern FILE *popen(const char*,const char*);
   extern int pclose(FILE*);
# else
#  define SQLITE_OMIT_POPEN 1
# endif
#endif

#if defined(_WIN32_WCE)
/* Windows CE (arm-wince-mingw32ce-gcc) does not provide isatty()
 * thus we always assume that we have a console. That can be
 * overridden with the -batch command line option.
 */
#define isatty(x) 1
#endif

/* ctype macros that work with signed characters */
#define IsSpace(X)  isspace((unsigned char)X)
#define IsDigit(X)  isdigit((unsigned char)X)
#define ToLower(X)  (char)tolower((unsigned char)X)

#if defined(_WIN32) || defined(WIN32)
#include <windows.h>

/* string conversion routines only needed on Win32 */
extern char *sqlite3_win32_unicode_to_utf8(LPCWSTR);
extern char *sqlite3_win32_mbcs_to_utf8_v2(const char *, int);
extern char *sqlite3_win32_utf8_to_mbcs_v2(const char *, int);
extern LPWSTR sqlite3_win32_utf8_to_unicode(const char *zText);
#endif

/* On Windows, we normally run with output mode of TEXT so that \n characters
** are automatically translated into \r\n.  However, this behavior needs
** to be disabled in some cases (ex: when generating CSV output and when
** rendering quoted strings that contain \n characters).  The following
** routines take care of that.
*/
#if defined(_WIN32) || defined(WIN32)
static void setBinaryMode(FILE *file, int isOutput){
  if( isOutput ) fflush(file);
  _setmode(_fileno(file), _O_BINARY);
}
static void setTextMode(FILE *file, int isOutput){
  if( isOutput ) fflush(file);
  _setmode(_fileno(file), _O_TEXT);
}
#else
# define setBinaryMode(X,Y)
# define setTextMode(X,Y)
#endif


/* True if the timer is enabled */
static int enableTimer = 0;

/* Return the current wall-clock time */
static sqlite3_int64 timeOfDay(void){
  static sqlite3_vfs *clockVfs = 0;
  sqlite3_int64 t;
  if( clockVfs==0 ) clockVfs = sqlite3_vfs_find(0);
  if( clockVfs->iVersion>=2 && clockVfs->xCurrentTimeInt64!=0 ){
    clockVfs->xCurrentTimeInt64(clockVfs, &t);
  }else{
    double r;
    clockVfs->xCurrentTime(clockVfs, &r);
    t = (sqlite3_int64)(r*86400000.0);
  }
  return t;
}

#if !defined(_WIN32) && !defined(WIN32) && !defined(__minux)
#include <sys/time.h>
#include <sys/resource.h>

/* VxWorks does not support getrusage() as far as we can determine */
#if defined(_WRS_KERNEL) || defined(__RTP__)
struct rusage {
  struct timeval ru_utime; /* user CPU time used */
  struct timeval ru_stime; /* system CPU time used */
};
#define getrusage(A,B) memset(B,0,sizeof(*B))
#endif

/* Saved resource information for the beginning of an operation */
static struct rusage sBegin;  /* CPU time at start */
static sqlite3_int64 iBegin;  /* Wall-clock time at start */

/*
** Begin timing an operation
*/
static void beginTimer(void){
  if( enableTimer ){
    getrusage(RUSAGE_SELF, &sBegin);
    iBegin = timeOfDay();
  }
}

/* Return the difference of two time_structs in seconds */
static double timeDiff(struct timeval *pStart, struct timeval *pEnd){
  return (pEnd->tv_usec - pStart->tv_usec)*0.000001 +
         (double)(pEnd->tv_sec - pStart->tv_sec);
}

/*
** Print the timing results.
*/
static void endTimer(void){
  if( enableTimer ){
    sqlite3_int64 iEnd = timeOfDay();
    struct rusage sEnd;
    getrusage(RUSAGE_SELF, &sEnd);
    printf("Run Time: real %.3f user %f sys %f\n",
       (iEnd - iBegin)*0.001,
       timeDiff(&sBegin.ru_utime, &sEnd.ru_utime),
       timeDiff(&sBegin.ru_stime, &sEnd.ru_stime));
  }
}

#define BEGIN_TIMER beginTimer()
#define END_TIMER endTimer()
#define HAS_TIMER 1

#elif (defined(_WIN32) || defined(WIN32))

/* Saved resource information for the beginning of an operation */
static HANDLE hProcess;
static FILETIME ftKernelBegin;
static FILETIME ftUserBegin;
static sqlite3_int64 ftWallBegin;
typedef BOOL (WINAPI *GETPROCTIMES)(HANDLE, LPFILETIME, LPFILETIME,
                                    LPFILETIME, LPFILETIME);
static GETPROCTIMES getProcessTimesAddr = NULL;

/*
** Check to see if we have timer support.  Return 1 if necessary
** support found (or found previously).
*/
static int hasTimer(void){
  if( getProcessTimesAddr ){
    return 1;
  } else {
    /* GetProcessTimes() isn't supported in WIN95 and some other Windows
    ** versions. See if the version we are running on has it, and if it
    ** does, save off a pointer to it and the current process handle.
    */
    hProcess = GetCurrentProcess();
    if( hProcess ){
      HINSTANCE hinstLib = LoadLibrary(TEXT("Kernel32.dll"));
      if( NULL != hinstLib ){
        getProcessTimesAddr =
            (GETPROCTIMES) GetProcAddress(hinstLib, "GetProcessTimes");
        if( NULL != getProcessTimesAddr ){
          return 1;
        }
        FreeLibrary(hinstLib);
      }
    }
  }
  return 0;
}

/*
** Begin timing an operation
*/
static void beginTimer(void){
  if( enableTimer && getProcessTimesAddr ){
    FILETIME ftCreation, ftExit;
    getProcessTimesAddr(hProcess,&ftCreation,&ftExit,
                        &ftKernelBegin,&ftUserBegin);
    ftWallBegin = timeOfDay();
  }
}

/* Return the difference of two FILETIME structs in seconds */
static double timeDiff(FILETIME *pStart, FILETIME *pEnd){
  sqlite_int64 i64Start = *((sqlite_int64 *) pStart);
  sqlite_int64 i64End = *((sqlite_int64 *) pEnd);
  return (double) ((i64End - i64Start) / 10000000.0);
}

/*
** Print the timing results.
*/
static void endTimer(void){
  if( enableTimer && getProcessTimesAddr){
    FILETIME ftCreation, ftExit, ftKernelEnd, ftUserEnd;
    sqlite3_int64 ftWallEnd = timeOfDay();
    getProcessTimesAddr(hProcess,&ftCreation,&ftExit,&ftKernelEnd,&ftUserEnd);
    printf("Run Time: real %.3f user %f sys %f\n",
       (ftWallEnd - ftWallBegin)*0.001,
       timeDiff(&ftUserBegin, &ftUserEnd),
       timeDiff(&ftKernelBegin, &ftKernelEnd));
  }
}

#define BEGIN_TIMER beginTimer()
#define END_TIMER endTimer()
#define HAS_TIMER hasTimer()

#else
#define BEGIN_TIMER
#define END_TIMER
#define HAS_TIMER 0
#endif

/*
** Used to prevent warnings about unused parameters
*/
#define UNUSED_PARAMETER(x) (void)(x)

/*
** If the following flag is set, then command execution stops
** at an error if we are not interactive.
*/
static int bail_on_error = 0;

/*
** Threat stdin as an interactive input if the following variable
** is true.  Otherwise, assume stdin is connected to a file or pipe.
*/
static int stdin_is_interactive = 1;

/*
** On Windows systems we have to know if standard output is a console
** in order to translate UTF-8 into MBCS.  The following variable is
** true if translation is required.
*/
static int stdout_is_console = 1;

/*
** The following is the open SQLite database.  We make a pointer
** to this database a static variable so that it can be accessed
** by the SIGINT handler to interrupt database processing.
*/
static sqlite3 *globalDb = 0;

/*
** True if an interrupt (Control-C) has been received.
*/
static volatile int seenInterrupt = 0;

/*
** This is the name of our program. It is set in main(), used
** in a number of other places, mostly for error messages.
*/
static char *Argv0;

/*
** Prompt strings. Initialized in main. Settable with
**   .prompt main continue
*/
static char mainPrompt[20];     /* First line prompt. default: "sqlite> "*/
static char continuePrompt[20]; /* Continuation prompt. default: "   ...> " */

/*
** Render output like fprintf().  Except, if the output is going to the
** console and if this is running on a Windows machine, translate the
** output from UTF-8 into MBCS.
*/
#if defined(_WIN32) || defined(WIN32)
void utf8_printf(FILE *out, const char *zFormat, ...){
  va_list ap;
  va_start(ap, zFormat);
  if( stdout_is_console && (out==stdout || out==stderr) ){
    char *z1 = sqlite3_vmprintf(zFormat, ap);
    char *z2 = sqlite3_win32_utf8_to_mbcs_v2(z1, 0);
    sqlite3_free(z1);
    fputs(z2, out);
    sqlite3_free(z2);
  }else{
    vfprintf(out, zFormat, ap);
  }
  va_end(ap);
}
#elif !defined(utf8_printf)
# define utf8_printf fprintf
#endif

/*
** Render output like fprintf().  This should not be used on anything that
** includes string formatting (e.g. "%s").
*/
#if !defined(raw_printf)
# define raw_printf fprintf
#endif

/*
** Write I/O traces to the following stream.
*/
#ifdef SQLITE_ENABLE_IOTRACE
static FILE *iotrace = 0;
#endif

/*
** This routine works like printf in that its first argument is a
** format string and subsequent arguments are values to be substituted
** in place of % fields.  The result of formatting this string
** is written to iotrace.
*/
#ifdef SQLITE_ENABLE_IOTRACE
static void SQLITE_CDECL iotracePrintf(const char *zFormat, ...){
  va_list ap;
  char *z;
  if( iotrace==0 ) return;
  va_start(ap, zFormat);
  z = sqlite3_vmprintf(zFormat, ap);
  va_end(ap);
  utf8_printf(iotrace, "%s", z);
  sqlite3_free(z);
}
#endif


/*
** Determines if a string is a number of not.
*/
static int isNumber(const char *z, int *realnum){
  if( *z=='-' || *z=='+' ) z++;
  if( !IsDigit(*z) ){
    return 0;
  }
  z++;
  if( realnum ) *realnum = 0;
  while( IsDigit(*z) ){ z++; }
  if( *z=='.' ){
    z++;
    if( !IsDigit(*z) ) return 0;
    while( IsDigit(*z) ){ z++; }
    if( realnum ) *realnum = 1;
  }
  if( *z=='e' || *z=='E' ){
    z++;
    if( *z=='+' || *z=='-' ) z++;
    if( !IsDigit(*z) ) return 0;
    while( IsDigit(*z) ){ z++; }
    if( realnum ) *realnum = 1;
  }
  return *z==0;
}

/*
** Compute a string length that is limited to what can be stored in
** lower 30 bits of a 32-bit signed integer.
*/
static int strlen30(const char *z){
  const char *z2 = z;
  while( *z2 ){ z2++; }
  return 0x3fffffff & (int)(z2 - z);
}

/*
** This routine reads a line of text from FILE in, stores
** the text in memory obtained from malloc() and returns a pointer
** to the text.  NULL is returned at end of file, or if malloc()
** fails.
**
** If zLine is not NULL then it is a malloced buffer returned from
** a previous call to this routine that may be reused.
*/
static char *local_getline(char *zLine, FILE *in){
  int nLine = zLine==0 ? 0 : 100;
  int n = 0;

  while( 1 ){
    if( n+100>nLine ){
      nLine = nLine*2 + 100;
      zLine = realloc(zLine, nLine);
      if( zLine==0 ) return 0;
    }
    if( fgets(&zLine[n], nLine - n, in)==0 ){
      if( n==0 ){
        free(zLine);
        return 0;
      }
      zLine[n] = 0;
      break;
    }
    while( zLine[n] ) n++;
    if( n>0 && zLine[n-1]=='\n' ){
      n--;
      if( n>0 && zLine[n-1]=='\r' ) n--;
      zLine[n] = 0;
      break;
    }
  }
#if defined(_WIN32) || defined(WIN32)
  /* For interactive input on Windows systems, translate the
  ** multi-byte characterset characters into UTF-8. */
  if( stdin_is_interactive && in==stdin ){
    char *zTrans = sqlite3_win32_mbcs_to_utf8_v2(zLine, 0);
    if( zTrans ){
      int nTrans = strlen30(zTrans)+1;
      if( nTrans>nLine ){
        zLine = realloc(zLine, nTrans);
        if( zLine==0 ){
          sqlite3_free(zTrans);
          return 0;
        }
      }
      memcpy(zLine, zTrans, nTrans);
      sqlite3_free(zTrans);
    }
  }
#endif /* defined(_WIN32) || defined(WIN32) */
  return zLine;
}

/*
** Retrieve a single line of input text.
**
** If in==0 then read from standard input and prompt before each line.
** If isContinuation is true, then a continuation prompt is appropriate.
** If isContinuation is zero, then the main prompt should be used.
**
** If zPrior is not NULL then it is a buffer from a prior call to this
** routine that can be reused.
**
** The result is stored in space obtained from malloc() and must either
** be freed by the caller or else passed back into this routine via the
** zPrior argument for reuse.
*/
static char *one_input_line(FILE *in, char *zPrior, int isContinuation){
  char *zPrompt;
  char *zResult;
  if( in!=0 ){
    zResult = local_getline(zPrior, in);
  }else{
    zPrompt = isContinuation ? continuePrompt : mainPrompt;
#if SHELL_USE_LOCAL_GETLINE
    printf("%s", zPrompt);
    fflush(stdout);
    zResult = local_getline(zPrior, stdin);
#else
    free(zPrior);
    zResult = shell_readline(zPrompt);
    if( zResult && *zResult ) shell_add_history(zResult);
#endif
  }
  return zResult;
}
/*
** A variable length string to which one can append text.
*/
typedef struct ShellText ShellText;
struct ShellText {
  char *z;
  int n;
  int nAlloc;
};

/*
** Initialize and destroy a ShellText object
*/
static void initText(ShellText *p){
  memset(p, 0, sizeof(*p));
}
static void freeText(ShellText *p){
  free(p->z);
  initText(p);
}

/* zIn is either a pointer to a NULL-terminated string in memory obtained
** from malloc(), or a NULL pointer. The string pointed to by zAppend is
** added to zIn, and the result returned in memory obtained from malloc().
** zIn, if it was not NULL, is freed.
**
** If the third argument, quote, is not '\0', then it is used as a
** quote character for zAppend.
*/
static void appendText(ShellText *p, char const *zAppend, char quote){
  int len;
  int i;
  int nAppend = strlen30(zAppend);

  len = nAppend+p->n+1;
  if( quote ){
    len += 2;
    for(i=0; i<nAppend; i++){
      if( zAppend[i]==quote ) len++;
    }
  }

  if( p->n+len>=p->nAlloc ){
    p->nAlloc = p->nAlloc*2 + len + 20;
    p->z = realloc(p->z, p->nAlloc);
    if( p->z==0 ){
      memset(p, 0, sizeof(*p));
      return;
    }
  }

  if( quote ){
    char *zCsr = p->z+p->n;
    *zCsr++ = quote;
    for(i=0; i<nAppend; i++){
      *zCsr++ = zAppend[i];
      if( zAppend[i]==quote ) *zCsr++ = quote;
    }
    *zCsr++ = quote;
    p->n = (int)(zCsr - p->z);
    *zCsr = '\0';
  }else{
    memcpy(p->z+p->n, zAppend, nAppend);
    p->n += nAppend;
    p->z[p->n] = '\0';
  }
}

/*
** Attempt to determine if identifier zName needs to be quoted, either
** because it contains non-alphanumeric characters, or because it is an
** SQLite keyword.  Be conservative in this estimate:  When in doubt assume
** that quoting is required.
**
** Return '"' if quoting is required.  Return 0 if no quoting is required.
*/
static char quoteChar(const char *zName){
  /* All SQLite keywords, in alphabetical order */
  static const char *azKeywords[] = {
    "ABORT", "ACTION", "ADD", "AFTER", "ALL", "ALTER", "ANALYZE", "AND", "AS",
    "ASC", "ATTACH", "AUTOINCREMENT", "BEFORE", "BEGIN", "BETWEEN", "BY",
    "CASCADE", "CASE", "CAST", "CHECK", "COLLATE", "COLUMN", "COMMIT",
    "CONFLICT", "CONSTRAINT", "CREATE", "CROSS", "CURRENT_DATE",
    "CURRENT_TIME", "CURRENT_TIMESTAMP", "DATABASE", "DEFAULT", "DEFERRABLE",
    "DEFERRED", "DELETE", "DESC", "DETACH", "DISTINCT", "DROP", "EACH",
    "ELSE", "END", "ESCAPE", "EXCEPT", "EXCLUSIVE", "EXISTS", "EXPLAIN",
    "FAIL", "FOR", "FOREIGN", "FROM", "FULL", "GLOB", "GROUP", "HAVING", "IF",
    "IGNORE", "IMMEDIATE", "IN", "INDEX", "INDEXED", "INITIALLY", "INNER",
    "INSERT", "INSTEAD", "INTERSECT", "INTO", "IS", "ISNULL", "JOIN", "KEY",
    "LEFT", "LIKE", "LIMIT", "MATCH", "NATURAL", "NO", "NOT", "NOTNULL",
    "NULL", "OF", "OFFSET", "ON", "OR", "ORDER", "OUTER", "PLAN", "PRAGMA",
    "PRIMARY", "QUERY", "RAISE", "RECURSIVE", "REFERENCES", "REGEXP",
    "REINDEX", "RELEASE", "RENAME", "REPLACE", "RESTRICT", "RIGHT",
    "ROLLBACK", "ROW", "SAVEPOINT", "SELECT", "SET", "TABLE", "TEMP",
    "TEMPORARY", "THEN", "TO", "TRANSACTION", "TRIGGER", "UNION", "UNIQUE",
    "UPDATE", "USING", "VACUUM", "VALUES", "VIEW", "VIRTUAL", "WHEN", "WHERE",
    "WITH", "WITHOUT",
  };
  int i, lwr, upr, mid, c;
  if( !isalpha((unsigned char)zName[0]) && zName[0]!='_' ) return '"';
  for(i=0; zName[i]; i++){
    if( !isalnum((unsigned char)zName[i]) && zName[i]!='_' ) return '"';
  }
  lwr = 0;
  upr = sizeof(azKeywords)/sizeof(azKeywords[0]) - 1;
  while( lwr<=upr ){
    mid = (lwr+upr)/2;
    c = sqlite3_stricmp(azKeywords[mid], zName);
    if( c==0 ) return '"';
    if( c<0 ){
      lwr = mid+1;
    }else{
      upr = mid-1;
    }
  }
  return 0;
}

/******************************************************************************
** SHA3 hash implementation copied from ../ext/misc/shathree.c
*/
typedef sqlite3_uint64 u64;
/*
** Macros to determine whether the machine is big or little endian,
** and whether or not that determination is run-time or compile-time.
**
** For best performance, an attempt is made to guess at the byte-order
** using C-preprocessor macros.  If that is unsuccessful, or if
** -DSHA3_BYTEORDER=0 is set, then byte-order is determined
** at run-time.
*/
#ifndef SHA3_BYTEORDER
# if defined(i386)     || defined(__i386__)   || defined(_M_IX86) ||    \
     defined(__x86_64) || defined(__x86_64__) || defined(_M_X64)  ||    \
     defined(_M_AMD64) || defined(_M_ARM)     || defined(__x86)   ||    \
     defined(__arm__)
#   define SHA3_BYTEORDER    1234
# elif defined(sparc)    || defined(__ppc__)
#   define SHA3_BYTEORDER    4321
# else
#   define SHA3_BYTEORDER 0
# endif
#endif


/*
** State structure for a SHA3 hash in progress
*/
typedef struct SHA3Context SHA3Context;
struct SHA3Context {
  union {
    u64 s[25];                /* Keccak state. 5x5 lines of 64 bits each */
    unsigned char x[1600];    /* ... or 1600 bytes */
  } u;
  unsigned nRate;        /* Bytes of input accepted per Keccak iteration */
  unsigned nLoaded;      /* Input bytes loaded into u.x[] so far this cycle */
  unsigned ixMask;       /* Insert next input into u.x[nLoaded^ixMask]. */
};

/*
** A single step of the Keccak mixing function for a 1600-bit state
*/
static void KeccakF1600Step(SHA3Context *p){
  int i;
  u64 B0, B1, B2, B3, B4;
  u64 C0, C1, C2, C3, C4;
  u64 D0, D1, D2, D3, D4;
  static const u64 RC[] = {
    0x0000000000000001ULL,  0x0000000000008082ULL,
    0x800000000000808aULL,  0x8000000080008000ULL,
    0x000000000000808bULL,  0x0000000080000001ULL,
    0x8000000080008081ULL,  0x8000000000008009ULL,
    0x000000000000008aULL,  0x0000000000000088ULL,
    0x0000000080008009ULL,  0x000000008000000aULL,
    0x000000008000808bULL,  0x800000000000008bULL,
    0x8000000000008089ULL,  0x8000000000008003ULL,
    0x8000000000008002ULL,  0x8000000000000080ULL,
    0x000000000000800aULL,  0x800000008000000aULL,
    0x8000000080008081ULL,  0x8000000000008080ULL,
    0x0000000080000001ULL,  0x8000000080008008ULL
  };
# define A00 (p->u.s[0])
# define A01 (p->u.s[1])
# define A02 (p->u.s[2])
# define A03 (p->u.s[3])
# define A04 (p->u.s[4])
# define A10 (p->u.s[5])
# define A11 (p->u.s[6])
# define A12 (p->u.s[7])
# define A13 (p->u.s[8])
# define A14 (p->u.s[9])
# define A20 (p->u.s[10])
# define A21 (p->u.s[11])
# define A22 (p->u.s[12])
# define A23 (p->u.s[13])
# define A24 (p->u.s[14])
# define A30 (p->u.s[15])
# define A31 (p->u.s[16])
# define A32 (p->u.s[17])
# define A33 (p->u.s[18])
# define A34 (p->u.s[19])
# define A40 (p->u.s[20])
# define A41 (p->u.s[21])
# define A42 (p->u.s[22])
# define A43 (p->u.s[23])
# define A44 (p->u.s[24])
# define ROL64(a,x) ((a<<x)|(a>>(64-x)))

  for(i=0; i<24; i+=4){
    C0 = A00^A10^A20^A30^A40;
    C1 = A01^A11^A21^A31^A41;
    C2 = A02^A12^A22^A32^A42;
    C3 = A03^A13^A23^A33^A43;
    C4 = A04^A14^A24^A34^A44;
    D0 = C4^ROL64(C1, 1);
    D1 = C0^ROL64(C2, 1);
    D2 = C1^ROL64(C3, 1);
    D3 = C2^ROL64(C4, 1);
    D4 = C3^ROL64(C0, 1);

    B0 = (A00^D0);
    B1 = ROL64((A11^D1), 44);
    B2 = ROL64((A22^D2), 43);
    B3 = ROL64((A33^D3), 21);
    B4 = ROL64((A44^D4), 14);
    A00 =   B0 ^((~B1)&  B2 );
    A00 ^= RC[i];
    A11 =   B1 ^((~B2)&  B3 );
    A22 =   B2 ^((~B3)&  B4 );
    A33 =   B3 ^((~B4)&  B0 );
    A44 =   B4 ^((~B0)&  B1 );

    B2 = ROL64((A20^D0), 3);
    B3 = ROL64((A31^D1), 45);
    B4 = ROL64((A42^D2), 61);
    B0 = ROL64((A03^D3), 28);
    B1 = ROL64((A14^D4), 20);
    A20 =   B0 ^((~B1)&  B2 );
    A31 =   B1 ^((~B2)&  B3 );
    A42 =   B2 ^((~B3)&  B4 );
    A03 =   B3 ^((~B4)&  B0 );
    A14 =   B4 ^((~B0)&  B1 );

    B4 = ROL64((A40^D0), 18);
    B0 = ROL64((A01^D1), 1);
    B1 = ROL64((A12^D2), 6);
    B2 = ROL64((A23^D3), 25);
    B3 = ROL64((A34^D4), 8);
    A40 =   B0 ^((~B1)&  B2 );
    A01 =   B1 ^((~B2)&  B3 );
    A12 =   B2 ^((~B3)&  B4 );
    A23 =   B3 ^((~B4)&  B0 );
    A34 =   B4 ^((~B0)&  B1 );

    B1 = ROL64((A10^D0), 36);
    B2 = ROL64((A21^D1), 10);
    B3 = ROL64((A32^D2), 15);
    B4 = ROL64((A43^D3), 56);
    B0 = ROL64((A04^D4), 27);
    A10 =   B0 ^((~B1)&  B2 );
    A21 =   B1 ^((~B2)&  B3 );
    A32 =   B2 ^((~B3)&  B4 );
    A43 =   B3 ^((~B4)&  B0 );
    A04 =   B4 ^((~B0)&  B1 );

    B3 = ROL64((A30^D0), 41);
    B4 = ROL64((A41^D1), 2);
    B0 = ROL64((A02^D2), 62);
    B1 = ROL64((A13^D3), 55);
    B2 = ROL64((A24^D4), 39);
    A30 =   B0 ^((~B1)&  B2 );
    A41 =   B1 ^((~B2)&  B3 );
    A02 =   B2 ^((~B3)&  B4 );
    A13 =   B3 ^((~B4)&  B0 );
    A24 =   B4 ^((~B0)&  B1 );

    C0 = A00^A20^A40^A10^A30;
    C1 = A11^A31^A01^A21^A41;
    C2 = A22^A42^A12^A32^A02;
    C3 = A33^A03^A23^A43^A13;
    C4 = A44^A14^A34^A04^A24;
    D0 = C4^ROL64(C1, 1);
    D1 = C0^ROL64(C2, 1);
    D2 = C1^ROL64(C3, 1);
    D3 = C2^ROL64(C4, 1);
    D4 = C3^ROL64(C0, 1);

    B0 = (A00^D0);
    B1 = ROL64((A31^D1), 44);
    B2 = ROL64((A12^D2), 43);
    B3 = ROL64((A43^D3), 21);
    B4 = ROL64((A24^D4), 14);
    A00 =   B0 ^((~B1)&  B2 );
    A00 ^= RC[i+1];
    A31 =   B1 ^((~B2)&  B3 );
    A12 =   B2 ^((~B3)&  B4 );
    A43 =   B3 ^((~B4)&  B0 );
    A24 =   B4 ^((~B0)&  B1 );

    B2 = ROL64((A40^D0), 3);
    B3 = ROL64((A21^D1), 45);
    B4 = ROL64((A02^D2), 61);
    B0 = ROL64((A33^D3), 28);
    B1 = ROL64((A14^D4), 20);
    A40 =   B0 ^((~B1)&  B2 );
    A21 =   B1 ^((~B2)&  B3 );
    A02 =   B2 ^((~B3)&  B4 );
    A33 =   B3 ^((~B4)&  B0 );
    A14 =   B4 ^((~B0)&  B1 );

    B4 = ROL64((A30^D0), 18);
    B0 = ROL64((A11^D1), 1);
    B1 = ROL64((A42^D2), 6);
    B2 = ROL64((A23^D3), 25);
    B3 = ROL64((A04^D4), 8);
    A30 =   B0 ^((~B1)&  B2 );
    A11 =   B1 ^((~B2)&  B3 );
    A42 =   B2 ^((~B3)&  B4 );
    A23 =   B3 ^((~B4)&  B0 );
    A04 =   B4 ^((~B0)&  B1 );

    B1 = ROL64((A20^D0), 36);
    B2 = ROL64((A01^D1), 10);
    B3 = ROL64((A32^D2), 15);
    B4 = ROL64((A13^D3), 56);
    B0 = ROL64((A44^D4), 27);
    A20 =   B0 ^((~B1)&  B2 );
    A01 =   B1 ^((~B2)&  B3 );
    A32 =   B2 ^((~B3)&  B4 );
    A13 =   B3 ^((~B4)&  B0 );
    A44 =   B4 ^((~B0)&  B1 );

    B3 = ROL64((A10^D0), 41);
    B4 = ROL64((A41^D1), 2);
    B0 = ROL64((A22^D2), 62);
    B1 = ROL64((A03^D3), 55);
    B2 = ROL64((A34^D4), 39);
    A10 =   B0 ^((~B1)&  B2 );
    A41 =   B1 ^((~B2)&  B3 );
    A22 =   B2 ^((~B3)&  B4 );
    A03 =   B3 ^((~B4)&  B0 );
    A34 =   B4 ^((~B0)&  B1 );

    C0 = A00^A40^A30^A20^A10;
    C1 = A31^A21^A11^A01^A41;
    C2 = A12^A02^A42^A32^A22;
    C3 = A43^A33^A23^A13^A03;
    C4 = A24^A14^A04^A44^A34;
    D0 = C4^ROL64(C1, 1);
    D1 = C0^ROL64(C2, 1);
    D2 = C1^ROL64(C3, 1);
    D3 = C2^ROL64(C4, 1);
    D4 = C3^ROL64(C0, 1);

    B0 = (A00^D0);
    B1 = ROL64((A21^D1), 44);
    B2 = ROL64((A42^D2), 43);
    B3 = ROL64((A13^D3), 21);
    B4 = ROL64((A34^D4), 14);
    A00 =   B0 ^((~B1)&  B2 );
    A00 ^= RC[i+2];
    A21 =   B1 ^((~B2)&  B3 );
    A42 =   B2 ^((~B3)&  B4 );
    A13 =   B3 ^((~B4)&  B0 );
    A34 =   B4 ^((~B0)&  B1 );

    B2 = ROL64((A30^D0), 3);
    B3 = ROL64((A01^D1), 45);
    B4 = ROL64((A22^D2), 61);
    B0 = ROL64((A43^D3), 28);
    B1 = ROL64((A14^D4), 20);
    A30 =   B0 ^((~B1)&  B2 );
    A01 =   B1 ^((~B2)&  B3 );
    A22 =   B2 ^((~B3)&  B4 );
    A43 =   B3 ^((~B4)&  B0 );
    A14 =   B4 ^((~B0)&  B1 );

    B4 = ROL64((A10^D0), 18);
    B0 = ROL64((A31^D1), 1);
    B1 = ROL64((A02^D2), 6);
    B2 = ROL64((A23^D3), 25);
    B3 = ROL64((A44^D4), 8);
    A10 =   B0 ^((~B1)&  B2 );
    A31 =   B1 ^((~B2)&  B3 );
    A02 =   B2 ^((~B3)&  B4 );
    A23 =   B3 ^((~B4)&  B0 );
    A44 =   B4 ^((~B0)&  B1 );

    B1 = ROL64((A40^D0), 36);
    B2 = ROL64((A11^D1), 10);
    B3 = ROL64((A32^D2), 15);
    B4 = ROL64((A03^D3), 56);
    B0 = ROL64((A24^D4), 27);
    A40 =   B0 ^((~B1)&  B2 );
    A11 =   B1 ^((~B2)&  B3 );
    A32 =   B2 ^((~B3)&  B4 );
    A03 =   B3 ^((~B4)&  B0 );
    A24 =   B4 ^((~B0)&  B1 );

    B3 = ROL64((A20^D0), 41);
    B4 = ROL64((A41^D1), 2);
    B0 = ROL64((A12^D2), 62);
    B1 = ROL64((A33^D3), 55);
    B2 = ROL64((A04^D4), 39);
    A20 =   B0 ^((~B1)&  B2 );
    A41 =   B1 ^((~B2)&  B3 );
    A12 =   B2 ^((~B3)&  B4 );
    A33 =   B3 ^((~B4)&  B0 );
    A04 =   B4 ^((~B0)&  B1 );

    C0 = A00^A30^A10^A40^A20;
    C1 = A21^A01^A31^A11^A41;
    C2 = A42^A22^A02^A32^A12;
    C3 = A13^A43^A23^A03^A33;
    C4 = A34^A14^A44^A24^A04;
    D0 = C4^ROL64(C1, 1);
    D1 = C0^ROL64(C2, 1);
    D2 = C1^ROL64(C3, 1);
    D3 = C2^ROL64(C4, 1);
    D4 = C3^ROL64(C0, 1);

    B0 = (A00^D0);
    B1 = ROL64((A01^D1), 44);
    B2 = ROL64((A02^D2), 43);
    B3 = ROL64((A03^D3), 21);
    B4 = ROL64((A04^D4), 14);
    A00 =   B0 ^((~B1)&  B2 );
    A00 ^= RC[i+3];
    A01 =   B1 ^((~B2)&  B3 );
    A02 =   B2 ^((~B3)&  B4 );
    A03 =   B3 ^((~B4)&  B0 );
    A04 =   B4 ^((~B0)&  B1 );

    B2 = ROL64((A10^D0), 3);
    B3 = ROL64((A11^D1), 45);
    B4 = ROL64((A12^D2), 61);
    B0 = ROL64((A13^D3), 28);
    B1 = ROL64((A14^D4), 20);
    A10 =   B0 ^((~B1)&  B2 );
    A11 =   B1 ^((~B2)&  B3 );
    A12 =   B2 ^((~B3)&  B4 );
    A13 =   B3 ^((~B4)&  B0 );
    A14 =   B4 ^((~B0)&  B1 );

    B4 = ROL64((A20^D0), 18);
    B0 = ROL64((A21^D1), 1);
    B1 = ROL64((A22^D2), 6);
    B2 = ROL64((A23^D3), 25);
    B3 = ROL64((A24^D4), 8);
    A20 =   B0 ^((~B1)&  B2 );
    A21 =   B1 ^((~B2)&  B3 );
    A22 =   B2 ^((~B3)&  B4 );
    A23 =   B3 ^((~B4)&  B0 );
    A24 =   B4 ^((~B0)&  B1 );

    B1 = ROL64((A30^D0), 36);
    B2 = ROL64((A31^D1), 10);
    B3 = ROL64((A32^D2), 15);
    B4 = ROL64((A33^D3), 56);
    B0 = ROL64((A34^D4), 27);
    A30 =   B0 ^((~B1)&  B2 );
    A31 =   B1 ^((~B2)&  B3 );
    A32 =   B2 ^((~B3)&  B4 );
    A33 =   B3 ^((~B4)&  B0 );
    A34 =   B4 ^((~B0)&  B1 );

    B3 = ROL64((A40^D0), 41);
    B4 = ROL64((A41^D1), 2);
    B0 = ROL64((A42^D2), 62);
    B1 = ROL64((A43^D3), 55);
    B2 = ROL64((A44^D4), 39);
    A40 =   B0 ^((~B1)&  B2 );
    A41 =   B1 ^((~B2)&  B3 );
    A42 =   B2 ^((~B3)&  B4 );
    A43 =   B3 ^((~B4)&  B0 );
    A44 =   B4 ^((~B0)&  B1 );
  }
}

/*
** Initialize a new hash.  iSize determines the size of the hash
** in bits and should be one of 224, 256, 384, or 512.  Or iSize
** can be zero to use the default hash size of 256 bits.
*/
static void SHA3Init(SHA3Context *p, int iSize){
  memset(p, 0, sizeof(*p));
  if( iSize>=128 && iSize<=512 ){
    p->nRate = (1600 - ((iSize + 31)&~31)*2)/8;
  }else{
    p->nRate = (1600 - 2*256)/8;
  }
#if SHA3_BYTEORDER==1234
  /* Known to be little-endian at compile-time. No-op */
#elif SHA3_BYTEORDER==4321
  p->ixMask = 7;  /* Big-endian */
#else
  {
    static unsigned int one = 1;
    if( 1==*(unsigned char*)&one ){
      /* Little endian.  No byte swapping. */
      p->ixMask = 0;
    }else{
      /* Big endian.  Byte swap. */
      p->ixMask = 7;
    }
  }
#endif
}

/*
** Make consecutive calls to the SHA3Update function to add new content
** to the hash
*/
static void SHA3Update(
  SHA3Context *p,
  const unsigned char *aData,
  unsigned int nData
){
  unsigned int i = 0;
#if SHA3_BYTEORDER==1234
  if( (p->nLoaded % 8)==0 && ((aData - (const unsigned char*)0)&7)==0 ){
    for(; i+7<nData; i+=8){
      p->u.s[p->nLoaded/8] ^= *(u64*)&aData[i];
      p->nLoaded += 8;
      if( p->nLoaded>=p->nRate ){
        KeccakF1600Step(p);
        p->nLoaded = 0;
      }
    }
  }
#endif
  for(; i<nData; i++){
#if SHA3_BYTEORDER==1234
    p->u.x[p->nLoaded] ^= aData[i];
#elif SHA3_BYTEORDER==4321
    p->u.x[p->nLoaded^0x07] ^= aData[i];
#else
    p->u.x[p->nLoaded^p->ixMask] ^= aData[i];
#endif
    p->nLoaded++;
    if( p->nLoaded==p->nRate ){
      KeccakF1600Step(p);
      p->nLoaded = 0;
    }
  }
}

/*
** After all content has been added, invoke SHA3Final() to compute
** the final hash.  The function returns a pointer to the binary
** hash value.
*/
static unsigned char *SHA3Final(SHA3Context *p){
  unsigned int i;
  if( p->nLoaded==p->nRate-1 ){
    const unsigned char c1 = 0x86;
    SHA3Update(p, &c1, 1);
  }else{
    const unsigned char c2 = 0x06;
    const unsigned char c3 = 0x80;
    SHA3Update(p, &c2, 1);
    p->nLoaded = p->nRate - 1;
    SHA3Update(p, &c3, 1);
  }
  for(i=0; i<p->nRate; i++){
    p->u.x[i+p->nRate] = p->u.x[i^p->ixMask];
  }
  return &p->u.x[p->nRate];
}

/*
** Implementation of the sha3(X,SIZE) function.
**
** Return a BLOB which is the SIZE-bit SHA3 hash of X.  The default
** size is 256.  If X is a BLOB, it is hashed as is.  
** For all other non-NULL types of input, X is converted into a UTF-8 string
** and the string is hashed without the trailing 0x00 terminator.  The hash
** of a NULL value is NULL.
*/
static void sha3Func(
  sqlite3_context *context,
  int argc,
  sqlite3_value **argv
){
  SHA3Context cx;
  int eType = sqlite3_value_type(argv[0]);
  int nByte = sqlite3_value_bytes(argv[0]);
  int iSize;
  if( argc==1 ){
    iSize = 256;
  }else{
    iSize = sqlite3_value_int(argv[1]);
    if( iSize!=224 && iSize!=256 && iSize!=384 && iSize!=512 ){
      sqlite3_result_error(context, "SHA3 size should be one of: 224 256 "
                                    "384 512", -1);
      return;
    }
  }
  if( eType==SQLITE_NULL ) return;
  SHA3Init(&cx, iSize);
  if( eType==SQLITE_BLOB ){
    SHA3Update(&cx, sqlite3_value_blob(argv[0]), nByte);
  }else{
    SHA3Update(&cx, sqlite3_value_text(argv[0]), nByte);
  }
  sqlite3_result_blob(context, SHA3Final(&cx), iSize/8, SQLITE_TRANSIENT);
}

/* Compute a string using sqlite3_vsnprintf() with a maximum length
** of 50 bytes and add it to the hash.
*/
static void hash_step_vformat(
  SHA3Context *p,                 /* Add content to this context */
  const char *zFormat,
  ...
){
  va_list ap;
  int n;
  char zBuf[50];
  va_start(ap, zFormat);
  sqlite3_vsnprintf(sizeof(zBuf),zBuf,zFormat,ap);
  va_end(ap);
  n = (int)strlen(zBuf);
  SHA3Update(p, (unsigned char*)zBuf, n);
}

/*
** Implementation of the sha3_query(SQL,SIZE) function.
**
** This function compiles and runs the SQL statement(s) given in the
** argument. The results are hashed using a SIZE-bit SHA3.  The default
** size is 256.
**
** The format of the byte stream that is hashed is summarized as follows:
**
**       S<n>:<sql>
**       R
**       N
**       I<int>
**       F<ieee-float>
**       B<size>:<bytes>
**       T<size>:<text>
**
** <sql> is the original SQL text for each statement run and <n> is
** the size of that text.  The SQL text is UTF-8.  A single R character
** occurs before the start of each row.  N means a NULL value.
** I mean an 8-byte little-endian integer <int>.  F is a floating point
** number with an 8-byte little-endian IEEE floating point value <ieee-float>.
** B means blobs of <size> bytes.  T means text rendered as <size>
** bytes of UTF-8.  The <n> and <size> values are expressed as an ASCII
** text integers.
**
** For each SQL statement in the X input, there is one S segment.  Each
** S segment is followed by zero or more R segments, one for each row in the
** result set.  After each R, there are one or more N, I, F, B, or T segments,
** one for each column in the result set.  Segments are concatentated directly
** with no delimiters of any kind.
*/
static void sha3QueryFunc(
  sqlite3_context *context,
  int argc,
  sqlite3_value **argv
){
  sqlite3 *db = sqlite3_context_db_handle(context);
  const char *zSql = (const char*)sqlite3_value_text(argv[0]);
  sqlite3_stmt *pStmt = 0;
  int nCol;                   /* Number of columns in the result set */
  int i;                      /* Loop counter */
  int rc;
  int n;
  const char *z;
  SHA3Context cx;
  int iSize;

  if( argc==1 ){
    iSize = 256;
  }else{
    iSize = sqlite3_value_int(argv[1]);
    if( iSize!=224 && iSize!=256 && iSize!=384 && iSize!=512 ){
      sqlite3_result_error(context, "SHA3 size should be one of: 224 256 "
                                    "384 512", -1);
      return;
    }
  }
  if( zSql==0 ) return;
  SHA3Init(&cx, iSize);
  while( zSql[0] ){
    rc = sqlite3_prepare_v2(db, zSql, -1, &pStmt, &zSql);
    if( rc ){
      char *zMsg = sqlite3_mprintf("error SQL statement [%s]: %s",
                                   zSql, sqlite3_errmsg(db));
      sqlite3_finalize(pStmt);
      sqlite3_result_error(context, zMsg, -1);
      sqlite3_free(zMsg);
      return;
    }
    if( !sqlite3_stmt_readonly(pStmt) ){
      char *zMsg = sqlite3_mprintf("non-query: [%s]", sqlite3_sql(pStmt));
      sqlite3_finalize(pStmt);
      sqlite3_result_error(context, zMsg, -1);
      sqlite3_free(zMsg);
      return;
    }
    nCol = sqlite3_column_count(pStmt);
    z = sqlite3_sql(pStmt);
    if( z==0 ){
      sqlite3_finalize(pStmt);
      continue;
    }
    n = (int)strlen(z);
    hash_step_vformat(&cx,"S%d:",n);
    SHA3Update(&cx,(unsigned char*)z,n);

    /* Compute a hash over the result of the query */
    while( SQLITE_ROW==sqlite3_step(pStmt) ){
      SHA3Update(&cx,(const unsigned char*)"R",1);
      for(i=0; i<nCol; i++){
        switch( sqlite3_column_type(pStmt,i) ){
          case SQLITE_NULL: {
            SHA3Update(&cx, (const unsigned char*)"N",1);
            break;
          }
          case SQLITE_INTEGER: {
            sqlite3_uint64 u;
            int j;
            unsigned char x[9];
            sqlite3_int64 v = sqlite3_column_int64(pStmt,i);
            memcpy(&u, &v, 8);
            for(j=8; j>=1; j--){
              x[j] = u & 0xff;
              u >>= 8;
            }
            x[0] = 'I';
            SHA3Update(&cx, x, 9);
            break;
          }
          case SQLITE_FLOAT: {
            sqlite3_uint64 u;
            int j;
            unsigned char x[9];
            double r = sqlite3_column_double(pStmt,i);
            memcpy(&u, &r, 8);
            for(j=8; j>=1; j--){
              x[j] = u & 0xff;
              u >>= 8;
            }
            x[0] = 'F';
            SHA3Update(&cx,x,9);
            break;
          }
          case SQLITE_TEXT: {
            int n2 = sqlite3_column_bytes(pStmt, i);
            const unsigned char *z2 = sqlite3_column_text(pStmt, i);
            hash_step_vformat(&cx,"T%d:",n2);
            SHA3Update(&cx, z2, n2);
            break;
          }
          case SQLITE_BLOB: {
            int n2 = sqlite3_column_bytes(pStmt, i);
            const unsigned char *z2 = sqlite3_column_blob(pStmt, i);
            hash_step_vformat(&cx,"B%d:",n2);
            SHA3Update(&cx, z2, n2);
            break;
          }
        }
      }
    }
    sqlite3_finalize(pStmt);
  }
  sqlite3_result_blob(context, SHA3Final(&cx), iSize/8, SQLITE_TRANSIENT);
}
/* End of SHA3 hashing logic copy/pasted from ../ext/misc/shathree.c
********************************************************************************/

#if defined(SQLITE_ENABLE_SESSION)
/*
** State information for a single open session
*/
typedef struct OpenSession OpenSession;
struct OpenSession {
  char *zName;             /* Symbolic name for this session */
  int nFilter;             /* Number of xFilter rejection GLOB patterns */
  char **azFilter;         /* Array of xFilter rejection GLOB patterns */
  sqlite3_session *p;      /* The open session */
};
#endif

/*
** Shell output mode information from before ".explain on",
** saved so that it can be restored by ".explain off"
*/
typedef struct SavedModeInfo SavedModeInfo;
struct SavedModeInfo {
  int valid;          /* Is there legit data in here? */
  int mode;           /* Mode prior to ".explain on" */
  int showHeader;     /* The ".header" setting prior to ".explain on" */
  int colWidth[100];  /* Column widths prior to ".explain on" */
};

/*
** State information about the database connection is contained in an
** instance of the following structure.
*/
typedef struct ShellState ShellState;
struct ShellState {
  sqlite3 *db;           /* The database */
  int autoExplain;       /* Automatically turn on .explain mode */
  int autoEQP;           /* Run EXPLAIN QUERY PLAN prior to seach SQL stmt */
  int statsOn;           /* True to display memory stats before each finalize */
  int scanstatsOn;       /* True to display scan stats before each finalize */
  int outCount;          /* Revert to stdout when reaching zero */
  int cnt;               /* Number of records displayed so far */
  FILE *out;             /* Write results here */
  FILE *traceOut;        /* Output for sqlite3_trace() */
  int nErr;              /* Number of errors seen */
  int mode;              /* An output mode setting */
  int cMode;             /* temporary output mode for the current query */
  int normalMode;        /* Output mode before ".explain on" */
  int writableSchema;    /* True if PRAGMA writable_schema=ON */
  int showHeader;        /* True to show column names in List or Column mode */
  int nCheck;            /* Number of ".check" commands run */
  unsigned shellFlgs;    /* Various flags */
  char *zDestTable;      /* Name of destination table when MODE_Insert */
  char zTestcase[30];    /* Name of current test case */
  char colSeparator[20]; /* Column separator character for several modes */
  char rowSeparator[20]; /* Row separator character for MODE_Ascii */
  int colWidth[100];     /* Requested width of each column when in column mode*/
  int actualWidth[100];  /* Actual width of each column */
  char nullValue[20];    /* The text to print when a NULL comes back from
                         ** the database */
  char outfile[FILENAME_MAX]; /* Filename for *out */
  const char *zDbFilename;    /* name of the database file */
  char *zFreeOnClose;         /* Filename to free when closing */
  const char *zVfs;           /* Name of VFS to use */
  sqlite3_stmt *pStmt;   /* Current statement if any. */
  FILE *pLog;            /* Write log output here */
  int *aiIndent;         /* Array of indents used in MODE_Explain */
  int nIndent;           /* Size of array aiIndent[] */
  int iIndent;           /* Index of current op in aiIndent[] */
#if defined(SQLITE_ENABLE_SESSION)
  int nSession;             /* Number of active sessions */
  OpenSession aSession[4];  /* Array of sessions.  [0] is in focus. */
#endif
};

/*
** These are the allowed shellFlgs values
*/
#define SHFLG_Scratch        0x00000001 /* The --scratch option is used */
#define SHFLG_Pagecache      0x00000002 /* The --pagecache option is used */
#define SHFLG_Lookaside      0x00000004 /* Lookaside memory is used */
#define SHFLG_Backslash      0x00000008 /* The --backslash option is used */
#define SHFLG_PreserveRowid  0x00000010 /* .dump preserves rowid values */
#define SHFLG_CountChanges   0x00000020 /* .changes setting */
#define SHFLG_Echo           0x00000040 /* .echo or --echo setting */

/*
** Macros for testing and setting shellFlgs
*/
#define ShellHasFlag(P,X)    (((P)->shellFlgs & (X))!=0)
#define ShellSetFlag(P,X)    ((P)->shellFlgs|=(X))
#define ShellClearFlag(P,X)  ((P)->shellFlgs&=(~(X)))

/*
** These are the allowed modes.
*/
#define MODE_Line     0  /* One column per line.  Blank line between records */
#define MODE_Column   1  /* One record per line in neat columns */
#define MODE_List     2  /* One record per line with a separator */
#define MODE_Semi     3  /* Same as MODE_List but append ";" to each line */
#define MODE_Html     4  /* Generate an XHTML table */
#define MODE_Insert   5  /* Generate SQL "insert" statements */
#define MODE_Quote    6  /* Quote values as for SQL */
#define MODE_Tcl      7  /* Generate ANSI-C or TCL quoted elements */
#define MODE_Csv      8  /* Quote strings, numbers are plain */
#define MODE_Explain  9  /* Like MODE_Column, but do not truncate data */
#define MODE_Ascii   10  /* Use ASCII unit and record separators (0x1F/0x1E) */
#define MODE_Pretty  11  /* Pretty-print schemas */

static const char *modeDescr[] = {
  "line",
  "column",
  "list",
  "semi",
  "html",
  "insert",
  "quote",
  "tcl",
  "csv",
  "explain",
  "ascii",
  "prettyprint",
};

/*
** These are the column/row/line separators used by the various
** import/export modes.
*/
#define SEP_Column    "|"
#define SEP_Row       "\n"
#define SEP_Tab       "\t"
#define SEP_Space     " "
#define SEP_Comma     ","
#define SEP_CrLf      "\r\n"
#define SEP_Unit      "\x1F"
#define SEP_Record    "\x1E"

/*
** Number of elements in an array
*/
#define ArraySize(X)  (int)(sizeof(X)/sizeof(X[0]))

/*
** A callback for the sqlite3_log() interface.
*/
static void shellLog(void *pArg, int iErrCode, const char *zMsg){
  ShellState *p = (ShellState*)pArg;
  if( p->pLog==0 ) return;
  utf8_printf(p->pLog, "(%d) %s\n", iErrCode, zMsg);
  fflush(p->pLog);
}

/*
** Output the given string as a hex-encoded blob (eg. X'1234' )
*/
static void output_hex_blob(FILE *out, const void *pBlob, int nBlob){
  int i;
  char *zBlob = (char *)pBlob;
  raw_printf(out,"X'");
  for(i=0; i<nBlob; i++){ raw_printf(out,"%02x",zBlob[i]&0xff); }
  raw_printf(out,"'");
}

/*
** Output the given string as a quoted string using SQL quoting conventions.
**
** The "\n" and "\r" characters are converted to char(10) and char(13)
** to prevent them from being transformed by end-of-line translators.
*/
static void output_quoted_string(FILE *out, const char *z){
  int i;
  char c;
  setBinaryMode(out, 1);
  for(i=0; (c = z[i])!=0 && c!='\'' && c!='\n' && c!='\r'; i++){}
  if( c==0 ){
    utf8_printf(out,"'%s'",z);
  }else{
    int inQuote = 0;
    int bStarted = 0;
    while( *z ){
      for(i=0; (c = z[i])!=0 && c!='\n' && c!='\r' && c!='\''; i++){}
      if( c=='\'' ) i++;
      if( i ){
        if( !inQuote ){
          if( bStarted ) raw_printf(out, "||");
          raw_printf(out, "'");
          inQuote = 1;
        }
        utf8_printf(out, "%.*s", i, z);
        z += i;
        bStarted = 1;
      }
      if( c=='\'' ){
        raw_printf(out, "'");
        continue;
      }
      if( inQuote ){
        raw_printf(out, "'");
        inQuote = 0;
      }
      if( c==0 ){
        break;
      }
      for(i=0; (c = z[i])=='\r' || c=='\n'; i++){
        if( bStarted ) raw_printf(out, "||");
        raw_printf(out, "char(%d)", c);
        bStarted = 1;
      }
      z += i;
    }
    if( inQuote ) raw_printf(out, "'");
  }
  setTextMode(out, 1);
}

/*
** Output the given string as a quoted according to C or TCL quoting rules.
*/
static void output_c_string(FILE *out, const char *z){
  unsigned int c;
  fputc('"', out);
  while( (c = *(z++))!=0 ){
    if( c=='\\' ){
      fputc(c, out);
      fputc(c, out);
    }else if( c=='"' ){
      fputc('\\', out);
      fputc('"', out);
    }else if( c=='\t' ){
      fputc('\\', out);
      fputc('t', out);
    }else if( c=='\n' ){
      fputc('\\', out);
      fputc('n', out);
    }else if( c=='\r' ){
      fputc('\\', out);
      fputc('r', out);
    }else if( !isprint(c&0xff) ){
      raw_printf(out, "\\%03o", c&0xff);
    }else{
      fputc(c, out);
    }
  }
  fputc('"', out);
}

/*
** Output the given string with characters that are special to
** HTML escaped.
*/
static void output_html_string(FILE *out, const char *z){
  int i;
  if( z==0 ) z = "";
  while( *z ){
    for(i=0;   z[i]
            && z[i]!='<'
            && z[i]!='&'
            && z[i]!='>'
            && z[i]!='\"'
            && z[i]!='\'';
        i++){}
    if( i>0 ){
      utf8_printf(out,"%.*s",i,z);
    }
    if( z[i]=='<' ){
      raw_printf(out,"&lt;");
    }else if( z[i]=='&' ){
      raw_printf(out,"&amp;");
    }else if( z[i]=='>' ){
      raw_printf(out,"&gt;");
    }else if( z[i]=='\"' ){
      raw_printf(out,"&quot;");
    }else if( z[i]=='\'' ){
      raw_printf(out,"&#39;");
    }else{
      break;
    }
    z += i + 1;
  }
}

/*
** If a field contains any character identified by a 1 in the following
** array, then the string must be quoted for CSV.
*/
static const char needCsvQuote[] = {
  1, 1, 1, 1, 1, 1, 1, 1,   1, 1, 1, 1, 1, 1, 1, 1,
  1, 1, 1, 1, 1, 1, 1, 1,   1, 1, 1, 1, 1, 1, 1, 1,
  1, 0, 1, 0, 0, 0, 0, 1,   0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0,   0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0,   0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0,   0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0,   0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0,   0, 0, 0, 0, 0, 0, 0, 1,
  1, 1, 1, 1, 1, 1, 1, 1,   1, 1, 1, 1, 1, 1, 1, 1,
  1, 1, 1, 1, 1, 1, 1, 1,   1, 1, 1, 1, 1, 1, 1, 1,
  1, 1, 1, 1, 1, 1, 1, 1,   1, 1, 1, 1, 1, 1, 1, 1,
  1, 1, 1, 1, 1, 1, 1, 1,   1, 1, 1, 1, 1, 1, 1, 1,
  1, 1, 1, 1, 1, 1, 1, 1,   1, 1, 1, 1, 1, 1, 1, 1,
  1, 1, 1, 1, 1, 1, 1, 1,   1, 1, 1, 1, 1, 1, 1, 1,
  1, 1, 1, 1, 1, 1, 1, 1,   1, 1, 1, 1, 1, 1, 1, 1,
  1, 1, 1, 1, 1, 1, 1, 1,   1, 1, 1, 1, 1, 1, 1, 1,
};

/*
** Output a single term of CSV.  Actually, p->colSeparator is used for
** the separator, which may or may not be a comma.  p->nullValue is
** the null value.  Strings are quoted if necessary.  The separator
** is only issued if bSep is true.
*/
static void output_csv(ShellState *p, const char *z, int bSep){
  FILE *out = p->out;
  if( z==0 ){
    utf8_printf(out,"%s",p->nullValue);
  }else{
    int i;
    int nSep = strlen30(p->colSeparator);
    for(i=0; z[i]; i++){
      if( needCsvQuote[((unsigned char*)z)[i]]
         || (z[i]==p->colSeparator[0] &&
             (nSep==1 || memcmp(z, p->colSeparator, nSep)==0)) ){
        i = 0;
        break;
      }
    }
    if( i==0 ){
      putc('"', out);
      for(i=0; z[i]; i++){
        if( z[i]=='"' ) putc('"', out);
        putc(z[i], out);
      }
      putc('"', out);
    }else{
      utf8_printf(out, "%s", z);
    }
  }
  if( bSep ){
    utf8_printf(p->out, "%s", p->colSeparator);
  }
}

#ifdef SIGINT
/*
** This routine runs when the user presses Ctrl-C
*/
static void interrupt_handler(int NotUsed){
  UNUSED_PARAMETER(NotUsed);
  seenInterrupt++;
  if( seenInterrupt>2 ) exit(1);
  if( globalDb ) sqlite3_interrupt(globalDb);
}
#endif

#ifndef SQLITE_OMIT_AUTHORIZATION
/*
** When the ".auth ON" is set, the following authorizer callback is
** invoked.  It always returns SQLITE_OK.
*/
static int shellAuth(
  void *pClientData,
  int op,
  const char *zA1,
  const char *zA2,
  const char *zA3,
  const char *zA4
){
  ShellState *p = (ShellState*)pClientData;
  static const char *azAction[] = { 0,
     "CREATE_INDEX",         "CREATE_TABLE",         "CREATE_TEMP_INDEX",
     "CREATE_TEMP_TABLE",    "CREATE_TEMP_TRIGGER",  "CREATE_TEMP_VIEW",
     "CREATE_TRIGGER",       "CREATE_VIEW",          "DELETE",
     "DROP_INDEX",           "DROP_TABLE",           "DROP_TEMP_INDEX",
     "DROP_TEMP_TABLE",      "DROP_TEMP_TRIGGER",    "DROP_TEMP_VIEW",
     "DROP_TRIGGER",         "DROP_VIEW",            "INSERT",
     "PRAGMA",               "READ",                 "SELECT",
     "TRANSACTION",          "UPDATE",               "ATTACH",
     "DETACH",               "ALTER_TABLE",          "REINDEX",
     "ANALYZE",              "CREATE_VTABLE",        "DROP_VTABLE",
     "FUNCTION",             "SAVEPOINT",            "RECURSIVE"
  };
  int i;
  const char *az[4];
  az[0] = zA1;
  az[1] = zA2;
  az[2] = zA3;
  az[3] = zA4;
  utf8_printf(p->out, "authorizer: %s", azAction[op]);
  for(i=0; i<4; i++){
    raw_printf(p->out, " ");
    if( az[i] ){
      output_c_string(p->out, az[i]);
    }else{
      raw_printf(p->out, "NULL");
    }
  }
  raw_printf(p->out, "\n");
  return SQLITE_OK;
}
#endif

/*
** Print a schema statement.  Part of MODE_Semi and MODE_Pretty output.
**
** This routine converts some CREATE TABLE statements for shadow tables
** in FTS3/4/5 into CREATE TABLE IF NOT EXISTS statements.
*/
static void printSchemaLine(FILE *out, const char *z, const char *zTail){
  if( sqlite3_strglob("CREATE TABLE ['\"]*", z)==0 ){
    utf8_printf(out, "CREATE TABLE IF NOT EXISTS %s%s", z+13, zTail);
  }else{
    utf8_printf(out, "%s%s", z, zTail);
  }
}
static void printSchemaLineN(FILE *out, char *z, int n, const char *zTail){
  char c = z[n];
  z[n] = 0;
  printSchemaLine(out, z, zTail);
  z[n] = c;
}

/*
** This is the callback routine that the shell
** invokes for each row of a query result.
*/
static int shell_callback(
  void *pArg,
  int nArg,        /* Number of result columns */
  char **azArg,    /* Text of each result column */
  char **azCol,    /* Column names */
  int *aiType      /* Column types */
){
  int i;
  ShellState *p = (ShellState*)pArg;

  switch( p->cMode ){
    case MODE_Line: {
      int w = 5;
      if( azArg==0 ) break;
      for(i=0; i<nArg; i++){
        int len = strlen30(azCol[i] ? azCol[i] : "");
        if( len>w ) w = len;
      }
      if( p->cnt++>0 ) utf8_printf(p->out, "%s", p->rowSeparator);
      for(i=0; i<nArg; i++){
        utf8_printf(p->out,"%*s = %s%s", w, azCol[i],
                azArg[i] ? azArg[i] : p->nullValue, p->rowSeparator);
      }
      break;
    }
    case MODE_Explain:
    case MODE_Column: {
      static const int aExplainWidths[] = {4, 13, 4, 4, 4, 13, 2, 13};
      const int *colWidth;
      int showHdr;
      char *rowSep;
      if( p->cMode==MODE_Column ){
        colWidth = p->colWidth;
        showHdr = p->showHeader;
        rowSep = p->rowSeparator;
      }else{
        colWidth = aExplainWidths;
        showHdr = 1;
        rowSep = SEP_Row;
      }
      if( p->cnt++==0 ){
        for(i=0; i<nArg; i++){
          int w, n;
          if( i<ArraySize(p->colWidth) ){
            w = colWidth[i];
          }else{
            w = 0;
          }
          if( w==0 ){
            w = strlen30(azCol[i] ? azCol[i] : "");
            if( w<10 ) w = 10;
            n = strlen30(azArg && azArg[i] ? azArg[i] : p->nullValue);
            if( w<n ) w = n;
          }
          if( i<ArraySize(p->actualWidth) ){
            p->actualWidth[i] = w;
          }
          if( showHdr ){
            if( w<0 ){
              utf8_printf(p->out,"%*.*s%s",-w,-w,azCol[i],
                      i==nArg-1 ? rowSep : "  ");
            }else{
              utf8_printf(p->out,"%-*.*s%s",w,w,azCol[i],
                      i==nArg-1 ? rowSep : "  ");
            }
          }
        }
        if( showHdr ){
          for(i=0; i<nArg; i++){
            int w;
            if( i<ArraySize(p->actualWidth) ){
               w = p->actualWidth[i];
               if( w<0 ) w = -w;
            }else{
               w = 10;
            }
            utf8_printf(p->out,"%-*.*s%s",w,w,
                   "----------------------------------------------------------"
                   "----------------------------------------------------------",
                    i==nArg-1 ? rowSep : "  ");
          }
        }
      }
      if( azArg==0 ) break;
      for(i=0; i<nArg; i++){
        int w;
        if( i<ArraySize(p->actualWidth) ){
           w = p->actualWidth[i];
        }else{
           w = 10;
        }
        if( p->cMode==MODE_Explain && azArg[i] && strlen30(azArg[i])>w ){
          w = strlen30(azArg[i]);
        }
        if( i==1 && p->aiIndent && p->pStmt ){
          if( p->iIndent<p->nIndent ){
            utf8_printf(p->out, "%*.s", p->aiIndent[p->iIndent], "");
          }
          p->iIndent++;
        }
        if( w<0 ){
          utf8_printf(p->out,"%*.*s%s",-w,-w,
              azArg[i] ? azArg[i] : p->nullValue,
              i==nArg-1 ? rowSep : "  ");
        }else{
          utf8_printf(p->out,"%-*.*s%s",w,w,
              azArg[i] ? azArg[i] : p->nullValue,
              i==nArg-1 ? rowSep : "  ");
        }
      }
      break;
    }
    case MODE_Semi: {   /* .schema and .fullschema output */
      printSchemaLine(p->out, azArg[0], ";\n");
      break;
    }
    case MODE_Pretty: {  /* .schema and .fullschema with --indent */
      char *z;
      int j;
      int nParen = 0;
      char cEnd = 0;
      char c;
      int nLine = 0;
      assert( nArg==1 );
      if( azArg[0]==0 ) break;
      if( sqlite3_strlike("CREATE VIEW%", azArg[0], 0)==0
       || sqlite3_strlike("CREATE TRIG%", azArg[0], 0)==0
      ){
        utf8_printf(p->out, "%s;\n", azArg[0]);
        break;
      }
      z = sqlite3_mprintf("%s", azArg[0]);
      j = 0;
      for(i=0; IsSpace(z[i]); i++){}
      for(; (c = z[i])!=0; i++){
        if( IsSpace(c) ){
          if( IsSpace(z[j-1]) || z[j-1]=='(' ) continue;
        }else if( (c=='(' || c==')') && j>0 && IsSpace(z[j-1]) ){
          j--;
        }
        z[j++] = c;
      }
      while( j>0 && IsSpace(z[j-1]) ){ j--; }
      z[j] = 0;
      if( strlen30(z)>=79 ){
        for(i=j=0; (c = z[i])!=0; i++){
          if( c==cEnd ){
            cEnd = 0;
          }else if( c=='"' || c=='\'' || c=='`' ){
            cEnd = c;
          }else if( c=='[' ){
            cEnd = ']';
          }else if( c=='(' ){
            nParen++;
          }else if( c==')' ){
            nParen--;
            if( nLine>0 && nParen==0 && j>0 ){
              printSchemaLineN(p->out, z, j, "\n");
              j = 0;
            }
          }
          z[j++] = c;
          if( nParen==1 && (c=='(' || c==',' || c=='\n') ){
            if( c=='\n' ) j--;
            printSchemaLineN(p->out, z, j, "\n  ");
            j = 0;
            nLine++;
            while( IsSpace(z[i+1]) ){ i++; }
          }
        }
        z[j] = 0;
      }
      printSchemaLine(p->out, z, ";\n");
      sqlite3_free(z);
      break;
    }
    case MODE_List: {
      if( p->cnt++==0 && p->showHeader ){
        for(i=0; i<nArg; i++){
          utf8_printf(p->out,"%s%s",azCol[i],
                  i==nArg-1 ? p->rowSeparator : p->colSeparator);
        }
      }
      if( azArg==0 ) break;
      for(i=0; i<nArg; i++){
        char *z = azArg[i];
        if( z==0 ) z = p->nullValue;
        utf8_printf(p->out, "%s", z);
        if( i<nArg-1 ){
          utf8_printf(p->out, "%s", p->colSeparator);
        }else{
          utf8_printf(p->out, "%s", p->rowSeparator);
        }
      }
      break;
    }
    case MODE_Html: {
      if( p->cnt++==0 && p->showHeader ){
        raw_printf(p->out,"<TR>");
        for(i=0; i<nArg; i++){
          raw_printf(p->out,"<TH>");
          output_html_string(p->out, azCol[i]);
          raw_printf(p->out,"</TH>\n");
        }
        raw_printf(p->out,"</TR>\n");
      }
      if( azArg==0 ) break;
      raw_printf(p->out,"<TR>");
      for(i=0; i<nArg; i++){
        raw_printf(p->out,"<TD>");
        output_html_string(p->out, azArg[i] ? azArg[i] : p->nullValue);
        raw_printf(p->out,"</TD>\n");
      }
      raw_printf(p->out,"</TR>\n");
      break;
    }
    case MODE_Tcl: {
      if( p->cnt++==0 && p->showHeader ){
        for(i=0; i<nArg; i++){
          output_c_string(p->out,azCol[i] ? azCol[i] : "");
          if(i<nArg-1) utf8_printf(p->out, "%s", p->colSeparator);
        }
        utf8_printf(p->out, "%s", p->rowSeparator);
      }
      if( azArg==0 ) break;
      for(i=0; i<nArg; i++){
        output_c_string(p->out, azArg[i] ? azArg[i] : p->nullValue);
        if(i<nArg-1) utf8_printf(p->out, "%s", p->colSeparator);
      }
      utf8_printf(p->out, "%s", p->rowSeparator);
      break;
    }
    case MODE_Csv: {
      setBinaryMode(p->out, 1);
      if( p->cnt++==0 && p->showHeader ){
        for(i=0; i<nArg; i++){
          output_csv(p, azCol[i] ? azCol[i] : "", i<nArg-1);
        }
        utf8_printf(p->out, "%s", p->rowSeparator);
      }
      if( nArg>0 ){
        for(i=0; i<nArg; i++){
          output_csv(p, azArg[i], i<nArg-1);
        }
        utf8_printf(p->out, "%s", p->rowSeparator);
      }
      setTextMode(p->out, 1);
      break;
    }
    case MODE_Quote:
    case MODE_Insert: {
      if( azArg==0 ) break;
      if( p->cMode==MODE_Insert ){
        utf8_printf(p->out,"INSERT INTO %s",p->zDestTable);
        if( p->showHeader ){
          raw_printf(p->out,"(");
          for(i=0; i<nArg; i++){
            char *zSep = i>0 ? ",": "";
            utf8_printf(p->out, "%s%s", zSep, azCol[i]);
          }
          raw_printf(p->out,")");
        }
        raw_printf(p->out," VALUES(");
      }else if( p->cnt==0 && p->showHeader ){
        for(i=0; i<nArg; i++){
          if( i>0 ) raw_printf(p->out, ",");
          output_quoted_string(p->out, azCol[i]);
        }
        raw_printf(p->out,"\n");
      }
      p->cnt++;
      for(i=0; i<nArg; i++){
        char *zSep = i>0 ? ",": "";
        if( (azArg[i]==0) || (aiType && aiType[i]==SQLITE_NULL) ){
          utf8_printf(p->out,"%sNULL",zSep);
        }else if( aiType && aiType[i]==SQLITE_TEXT ){
          if( zSep[0] ) utf8_printf(p->out,"%s",zSep);
          output_quoted_string(p->out, azArg[i]);
        }else if( aiType && aiType[i]==SQLITE_INTEGER ){
          utf8_printf(p->out,"%s%s",zSep, azArg[i]);
        }else if( aiType && aiType[i]==SQLITE_FLOAT ){
          char z[50];
          double r = sqlite3_column_double(p->pStmt, i);
          sqlite3_snprintf(50,z,"%!.20g", r);
          raw_printf(p->out, "%s%s", zSep, z);
        }else if( aiType && aiType[i]==SQLITE_BLOB && p->pStmt ){
          const void *pBlob = sqlite3_column_blob(p->pStmt, i);
          int nBlob = sqlite3_column_bytes(p->pStmt, i);
          if( zSep[0] ) utf8_printf(p->out,"%s",zSep);
          output_hex_blob(p->out, pBlob, nBlob);
        }else if( isNumber(azArg[i], 0) ){
          utf8_printf(p->out,"%s%s",zSep, azArg[i]);
        }else{
          if( zSep[0] ) utf8_printf(p->out,"%s",zSep);
          output_quoted_string(p->out, azArg[i]);
        }
      }
      raw_printf(p->out,p->cMode==MODE_Quote?"\n":");\n");
      break;
    }
    case MODE_Ascii: {
      if( p->cnt++==0 && p->showHeader ){
        for(i=0; i<nArg; i++){
          if( i>0 ) utf8_printf(p->out, "%s", p->colSeparator);
          utf8_printf(p->out,"%s",azCol[i] ? azCol[i] : "");
        }
        utf8_printf(p->out, "%s", p->rowSeparator);
      }
      if( azArg==0 ) break;
      for(i=0; i<nArg; i++){
        if( i>0 ) utf8_printf(p->out, "%s", p->colSeparator);
        utf8_printf(p->out,"%s",azArg[i] ? azArg[i] : p->nullValue);
      }
      utf8_printf(p->out, "%s", p->rowSeparator);
      break;
    }
  }
  return 0;
}

/*
** This is the callback routine that the SQLite library
** invokes for each row of a query result.
*/
static int callback(void *pArg, int nArg, char **azArg, char **azCol){
  /* since we don't have type info, call the shell_callback with a NULL value */
  return shell_callback(pArg, nArg, azArg, azCol, NULL);
}

/*
** This is the callback routine from sqlite3_exec() that appends all
** output onto the end of a ShellText object.
*/
static int captureOutputCallback(void *pArg, int nArg, char **azArg, char **az){
  ShellText *p = (ShellText*)pArg;
  int i;
  UNUSED_PARAMETER(az);
  if( p->n ) appendText(p, "|", 0);
  for(i=0; i<nArg; i++){
    if( i ) appendText(p, ",", 0);
    if( azArg[i] ) appendText(p, azArg[i], 0);
  }
  return 0;
}

/*
** Generate an appropriate SELFTEST table in the main database.
*/
static void createSelftestTable(ShellState *p){
  char *zErrMsg = 0;
  sqlite3_exec(p->db,
    "SAVEPOINT selftest_init;\n"
    "CREATE TABLE IF NOT EXISTS selftest(\n"
    "  tno INTEGER PRIMARY KEY,\n"   /* Test number */
    "  op TEXT,\n"                   /* Operator:  memo run */
    "  cmd TEXT,\n"                  /* Command text */
    "  ans TEXT\n"                   /* Desired answer */
    ");"
    "CREATE TEMP TABLE [_shell$self](op,cmd,ans);\n"
    "INSERT INTO [_shell$self](rowid,op,cmd)\n"
    "  VALUES(coalesce((SELECT (max(tno)+100)/10 FROM selftest),10),\n"
    "         'memo','Tests generated by --init');\n"
    "INSERT INTO [_shell$self]\n"
    "  SELECT 'run',\n"
    "    'SELECT hex(sha3_query(''SELECT type,name,tbl_name,sql "
                                 "FROM sqlite_master ORDER BY 2'',224))',\n"
    "    hex(sha3_query('SELECT type,name,tbl_name,sql "
                          "FROM sqlite_master ORDER BY 2',224));\n"
    "INSERT INTO [_shell$self]\n"
    "  SELECT 'run',"
    "    'SELECT hex(sha3_query(''SELECT * FROM \"' ||"
    "        printf('%w',name) || '\" NOT INDEXED'',224))',\n"
    "    hex(sha3_query(printf('SELECT * FROM \"%w\" NOT INDEXED',name),224))\n"
    "  FROM (\n"
    "    SELECT name FROM sqlite_master\n"
    "     WHERE type='table'\n"
    "       AND name<>'selftest'\n"
    "       AND coalesce(rootpage,0)>0\n"
    "  )\n"
    " ORDER BY name;\n"
    "INSERT INTO [_shell$self]\n"
    "  VALUES('run','PRAGMA integrity_check','ok');\n"
    "INSERT INTO selftest(tno,op,cmd,ans)"
    "  SELECT rowid*10,op,cmd,ans FROM [_shell$self];\n"
    "DROP TABLE [_shell$self];"
    ,0,0,&zErrMsg);
  if( zErrMsg ){
    utf8_printf(stderr, "SELFTEST initialization failure: %s\n", zErrMsg);
    sqlite3_free(zErrMsg);
  }
  sqlite3_exec(p->db, "RELEASE selftest_init",0,0,0);
}


/*
** Set the destination table field of the ShellState structure to
** the name of the table given.  Escape any quote characters in the
** table name.
*/
static void set_table_name(ShellState *p, const char *zName){
  int i, n;
  int cQuote;
  char *z;

  if( p->zDestTable ){
    free(p->zDestTable);
    p->zDestTable = 0;
  }
  if( zName==0 ) return;
  cQuote = quoteChar(zName);
  n = strlen30(zName);
  if( cQuote ) n += 2;
  z = p->zDestTable = malloc( n+1 );
  if( z==0 ){
    raw_printf(stderr,"Error: out of memory\n");
    exit(1);
  }
  n = 0;
  if( cQuote ) z[n++] = cQuote;
  for(i=0; zName[i]; i++){
    z[n++] = zName[i];
    if( zName[i]==cQuote ) z[n++] = cQuote;
  }
  if( cQuote ) z[n++] = cQuote;
  z[n] = 0;
}


/*
** Execute a query statement that will generate SQL output.  Print
** the result columns, comma-separated, on a line and then add a
** semicolon terminator to the end of that line.
**
** If the number of columns is 1 and that column contains text "--"
** then write the semicolon on a separate line.  That way, if a
** "--" comment occurs at the end of the statement, the comment
** won't consume the semicolon terminator.
*/
static int run_table_dump_query(
  ShellState *p,           /* Query context */
  const char *zSelect,     /* SELECT statement to extract content */
  const char *zFirstRow    /* Print before first row, if not NULL */
){
  sqlite3_stmt *pSelect;
  int rc;
  int nResult;
  int i;
  const char *z;
  rc = sqlite3_prepare_v2(p->db, zSelect, -1, &pSelect, 0);
  if( rc!=SQLITE_OK || !pSelect ){
    utf8_printf(p->out, "/**** ERROR: (%d) %s *****/\n", rc,
                sqlite3_errmsg(p->db));
    if( (rc&0xff)!=SQLITE_CORRUPT ) p->nErr++;
    return rc;
  }
  rc = sqlite3_step(pSelect);
  nResult = sqlite3_column_count(pSelect);
  while( rc==SQLITE_ROW ){
    if( zFirstRow ){
      utf8_printf(p->out, "%s", zFirstRow);
      zFirstRow = 0;
    }
    z = (const char*)sqlite3_column_text(pSelect, 0);
    utf8_printf(p->out, "%s", z);
    for(i=1; i<nResult; i++){
      utf8_printf(p->out, ",%s", sqlite3_column_text(pSelect, i));
    }
    if( z==0 ) z = "";
    while( z[0] && (z[0]!='-' || z[1]!='-') ) z++;
    if( z[0] ){
      raw_printf(p->out, "\n;\n");
    }else{
      raw_printf(p->out, ";\n");
    }
    rc = sqlite3_step(pSelect);
  }
  rc = sqlite3_finalize(pSelect);
  if( rc!=SQLITE_OK ){
    utf8_printf(p->out, "/**** ERROR: (%d) %s *****/\n", rc,
                sqlite3_errmsg(p->db));
    if( (rc&0xff)!=SQLITE_CORRUPT ) p->nErr++;
  }
  return rc;
}

/*
** Allocate space and save off current error string.
*/
static char *save_err_msg(
  sqlite3 *db            /* Database to query */
){
  int nErrMsg = 1+strlen30(sqlite3_errmsg(db));
  char *zErrMsg = sqlite3_malloc64(nErrMsg);
  if( zErrMsg ){
    memcpy(zErrMsg, sqlite3_errmsg(db), nErrMsg);
  }
  return zErrMsg;
}

#ifdef __linux__
/*
** Attempt to display I/O stats on Linux using /proc/PID/io
*/
static void displayLinuxIoStats(FILE *out){
  FILE *in;
  char z[200];
  sqlite3_snprintf(sizeof(z), z, "/proc/%d/io", getpid());
  in = fopen(z, "rb");
  if( in==0 ) return;
  while( fgets(z, sizeof(z), in)!=0 ){
    static const struct {
      const char *zPattern;
      const char *zDesc;
    } aTrans[] = {
      { "rchar: ",                  "Bytes received by read():" },
      { "wchar: ",                  "Bytes sent to write():"    },
      { "syscr: ",                  "Read() system calls:"      },
      { "syscw: ",                  "Write() system calls:"     },
      { "read_bytes: ",             "Bytes read from storage:"  },
      { "write_bytes: ",            "Bytes written to storage:" },
      { "cancelled_write_bytes: ",  "Cancelled write bytes:"    },
    };
    int i;
    for(i=0; i<ArraySize(aTrans); i++){
      int n = (int)strlen(aTrans[i].zPattern);
      if( strncmp(aTrans[i].zPattern, z, n)==0 ){
        utf8_printf(out, "%-36s %s", aTrans[i].zDesc, &z[n]);
        break;
      }
    }
  }
  fclose(in);
}
#endif

/*
** Display a single line of status using 64-bit values.
*/
static void displayStatLine(
  ShellState *p,            /* The shell context */
  char *zLabel,             /* Label for this one line */
  char *zFormat,            /* Format for the result */
  int iStatusCtrl,          /* Which status to display */
  int bReset                /* True to reset the stats */
){
  sqlite3_int64 iCur = -1;
  sqlite3_int64 iHiwtr = -1;
  int i, nPercent;
  char zLine[200];
  sqlite3_status64(iStatusCtrl, &iCur, &iHiwtr, bReset);
  for(i=0, nPercent=0; zFormat[i]; i++){
    if( zFormat[i]=='%' ) nPercent++;
  }
  if( nPercent>1 ){
    sqlite3_snprintf(sizeof(zLine), zLine, zFormat, iCur, iHiwtr);
  }else{
    sqlite3_snprintf(sizeof(zLine), zLine, zFormat, iHiwtr);
  }
  raw_printf(p->out, "%-36s %s\n", zLabel, zLine);
}

/*
** Display memory stats.
*/
static int display_stats(
  sqlite3 *db,                /* Database to query */
  ShellState *pArg,           /* Pointer to ShellState */
  int bReset                  /* True to reset the stats */
){
  int iCur;
  int iHiwtr;

  if( pArg && pArg->out ){
    displayStatLine(pArg, "Memory Used:",
       "%lld (max %lld) bytes", SQLITE_STATUS_MEMORY_USED, bReset);
    displayStatLine(pArg, "Number of Outstanding Allocations:",
       "%lld (max %lld)", SQLITE_STATUS_MALLOC_COUNT, bReset);
    if( pArg->shellFlgs & SHFLG_Pagecache ){
      displayStatLine(pArg, "Number of Pcache Pages Used:",
         "%lld (max %lld) pages", SQLITE_STATUS_PAGECACHE_USED, bReset);
    }
    displayStatLine(pArg, "Number of Pcache Overflow Bytes:",
       "%lld (max %lld) bytes", SQLITE_STATUS_PAGECACHE_OVERFLOW, bReset);
    if( pArg->shellFlgs & SHFLG_Scratch ){
      displayStatLine(pArg, "Number of Scratch Allocations Used:",
         "%lld (max %lld)", SQLITE_STATUS_SCRATCH_USED, bReset);
    }
    displayStatLine(pArg, "Number of Scratch Overflow Bytes:",
       "%lld (max %lld) bytes", SQLITE_STATUS_SCRATCH_OVERFLOW, bReset);
    displayStatLine(pArg, "Largest Allocation:",
       "%lld bytes", SQLITE_STATUS_MALLOC_SIZE, bReset);
    displayStatLine(pArg, "Largest Pcache Allocation:",
       "%lld bytes", SQLITE_STATUS_PAGECACHE_SIZE, bReset);
    displayStatLine(pArg, "Largest Scratch Allocation:",
       "%lld bytes", SQLITE_STATUS_SCRATCH_SIZE, bReset);
#ifdef YYTRACKMAXSTACKDEPTH
    displayStatLine(pArg, "Deepest Parser Stack:",
       "%lld (max %lld)", SQLITE_STATUS_PARSER_STACK, bReset);
#endif
  }

  if( pArg && pArg->out && db ){
    if( pArg->shellFlgs & SHFLG_Lookaside ){
      iHiwtr = iCur = -1;
      sqlite3_db_status(db, SQLITE_DBSTATUS_LOOKASIDE_USED,
                        &iCur, &iHiwtr, bReset);
      raw_printf(pArg->out,
              "Lookaside Slots Used:                %d (max %d)\n",
              iCur, iHiwtr);
      sqlite3_db_status(db, SQLITE_DBSTATUS_LOOKASIDE_HIT,
                        &iCur, &iHiwtr, bReset);
      raw_printf(pArg->out, "Successful lookaside attempts:       %d\n",
              iHiwtr);
      sqlite3_db_status(db, SQLITE_DBSTATUS_LOOKASIDE_MISS_SIZE,
                        &iCur, &iHiwtr, bReset);
      raw_printf(pArg->out, "Lookaside failures due to size:      %d\n",
              iHiwtr);
      sqlite3_db_status(db, SQLITE_DBSTATUS_LOOKASIDE_MISS_FULL,
                        &iCur, &iHiwtr, bReset);
      raw_printf(pArg->out, "Lookaside failures due to OOM:       %d\n",
              iHiwtr);
    }
    iHiwtr = iCur = -1;
    sqlite3_db_status(db, SQLITE_DBSTATUS_CACHE_USED, &iCur, &iHiwtr, bReset);
    raw_printf(pArg->out, "Pager Heap Usage:                    %d bytes\n",
            iCur);
    iHiwtr = iCur = -1;
    sqlite3_db_status(db, SQLITE_DBSTATUS_CACHE_HIT, &iCur, &iHiwtr, 1);
    raw_printf(pArg->out, "Page cache hits:                     %d\n", iCur);
    iHiwtr = iCur = -1;
    sqlite3_db_status(db, SQLITE_DBSTATUS_CACHE_MISS, &iCur, &iHiwtr, 1);
    raw_printf(pArg->out, "Page cache misses:                   %d\n", iCur);
    iHiwtr = iCur = -1;
    sqlite3_db_status(db, SQLITE_DBSTATUS_CACHE_WRITE, &iCur, &iHiwtr, 1);
    raw_printf(pArg->out, "Page cache writes:                   %d\n", iCur);
    iHiwtr = iCur = -1;
    sqlite3_db_status(db, SQLITE_DBSTATUS_SCHEMA_USED, &iCur, &iHiwtr, bReset);
    raw_printf(pArg->out, "Schema Heap Usage:                   %d bytes\n",
            iCur);
    iHiwtr = iCur = -1;
    sqlite3_db_status(db, SQLITE_DBSTATUS_STMT_USED, &iCur, &iHiwtr, bReset);
    raw_printf(pArg->out, "Statement Heap/Lookaside Usage:      %d bytes\n",
            iCur);
  }

  if( pArg && pArg->out && db && pArg->pStmt ){
    iCur = sqlite3_stmt_status(pArg->pStmt, SQLITE_STMTSTATUS_FULLSCAN_STEP,
                               bReset);
    raw_printf(pArg->out, "Fullscan Steps:                      %d\n", iCur);
    iCur = sqlite3_stmt_status(pArg->pStmt, SQLITE_STMTSTATUS_SORT, bReset);
    raw_printf(pArg->out, "Sort Operations:                     %d\n", iCur);
    iCur = sqlite3_stmt_status(pArg->pStmt, SQLITE_STMTSTATUS_AUTOINDEX,bReset);
    raw_printf(pArg->out, "Autoindex Inserts:                   %d\n", iCur);
    iCur = sqlite3_stmt_status(pArg->pStmt, SQLITE_STMTSTATUS_VM_STEP, bReset);
    raw_printf(pArg->out, "Virtual Machine Steps:               %d\n", iCur);
  }

#ifdef __linux__
  displayLinuxIoStats(pArg->out);
#endif

  /* Do not remove this machine readable comment: extra-stats-output-here */

  return 0;
}

/*
** Display scan stats.
*/
static void display_scanstats(
  sqlite3 *db,                    /* Database to query */
  ShellState *pArg                /* Pointer to ShellState */
){
#ifndef SQLITE_ENABLE_STMT_SCANSTATUS
  UNUSED_PARAMETER(db);
  UNUSED_PARAMETER(pArg);
#else
  int i, k, n, mx;
  raw_printf(pArg->out, "-------- scanstats --------\n");
  mx = 0;
  for(k=0; k<=mx; k++){
    double rEstLoop = 1.0;
    for(i=n=0; 1; i++){
      sqlite3_stmt *p = pArg->pStmt;
      sqlite3_int64 nLoop, nVisit;
      double rEst;
      int iSid;
      const char *zExplain;
      if( sqlite3_stmt_scanstatus(p, i, SQLITE_SCANSTAT_NLOOP, (void*)&nLoop) ){
        break;
      }
      sqlite3_stmt_scanstatus(p, i, SQLITE_SCANSTAT_SELECTID, (void*)&iSid);
      if( iSid>mx ) mx = iSid;
      if( iSid!=k ) continue;
      if( n==0 ){
        rEstLoop = (double)nLoop;
        if( k>0 ) raw_printf(pArg->out, "-------- subquery %d -------\n", k);
      }
      n++;
      sqlite3_stmt_scanstatus(p, i, SQLITE_SCANSTAT_NVISIT, (void*)&nVisit);
      sqlite3_stmt_scanstatus(p, i, SQLITE_SCANSTAT_EST, (void*)&rEst);
      sqlite3_stmt_scanstatus(p, i, SQLITE_SCANSTAT_EXPLAIN, (void*)&zExplain);
      utf8_printf(pArg->out, "Loop %2d: %s\n", n, zExplain);
      rEstLoop *= rEst;
      raw_printf(pArg->out,
          "         nLoop=%-8lld nRow=%-8lld estRow=%-8lld estRow/Loop=%-8g\n",
          nLoop, nVisit, (sqlite3_int64)(rEstLoop+0.5), rEst
      );
    }
  }
  raw_printf(pArg->out, "---------------------------\n");
#endif
}

/*
** Parameter azArray points to a zero-terminated array of strings. zStr
** points to a single nul-terminated string. Return non-zero if zStr
** is equal, according to strcmp(), to any of the strings in the array.
** Otherwise, return zero.
*/
static int str_in_array(const char *zStr, const char **azArray){
  int i;
  for(i=0; azArray[i]; i++){
    if( 0==strcmp(zStr, azArray[i]) ) return 1;
  }
  return 0;
}

/*
** If compiled statement pSql appears to be an EXPLAIN statement, allocate
** and populate the ShellState.aiIndent[] array with the number of
** spaces each opcode should be indented before it is output.
**
** The indenting rules are:
**
**     * For each "Next", "Prev", "VNext" or "VPrev" instruction, indent
**       all opcodes that occur between the p2 jump destination and the opcode
**       itself by 2 spaces.
**
**     * For each "Goto", if the jump destination is earlier in the program
**       and ends on one of:
**          Yield  SeekGt  SeekLt  RowSetRead  Rewind
**       or if the P1 parameter is one instead of zero,
**       then indent all opcodes between the earlier instruction
**       and "Goto" by 2 spaces.
*/
static void explain_data_prepare(ShellState *p, sqlite3_stmt *pSql){
  const char *zSql;               /* The text of the SQL statement */
  const char *z;                  /* Used to check if this is an EXPLAIN */
  int *abYield = 0;               /* True if op is an OP_Yield */
  int nAlloc = 0;                 /* Allocated size of p->aiIndent[], abYield */
  int iOp;                        /* Index of operation in p->aiIndent[] */

  const char *azNext[] = { "Next", "Prev", "VPrev", "VNext", "SorterNext",
                           "NextIfOpen", "PrevIfOpen", 0 };
  const char *azYield[] = { "Yield", "SeekLT", "SeekGT", "RowSetRead",
                            "Rewind", 0 };
  const char *azGoto[] = { "Goto", 0 };

  /* Try to figure out if this is really an EXPLAIN statement. If this
  ** cannot be verified, return early.  */
  if( sqlite3_column_count(pSql)!=8 ){
    p->cMode = p->mode;
    return;
  }
  zSql = sqlite3_sql(pSql);
  if( zSql==0 ) return;
  for(z=zSql; *z==' ' || *z=='\t' || *z=='\n' || *z=='\f' || *z=='\r'; z++);
  if( sqlite3_strnicmp(z, "explain", 7) ){
    p->cMode = p->mode;
    return;
  }

  for(iOp=0; SQLITE_ROW==sqlite3_step(pSql); iOp++){
    int i;
    int iAddr = sqlite3_column_int(pSql, 0);
    const char *zOp = (const char*)sqlite3_column_text(pSql, 1);

    /* Set p2 to the P2 field of the current opcode. Then, assuming that
    ** p2 is an instruction address, set variable p2op to the index of that
    ** instruction in the aiIndent[] array. p2 and p2op may be different if
    ** the current instruction is part of a sub-program generated by an
    ** SQL trigger or foreign key.  */
    int p2 = sqlite3_column_int(pSql, 3);
    int p2op = (p2 + (iOp-iAddr));

    /* Grow the p->aiIndent array as required */
    if( iOp>=nAlloc ){
      if( iOp==0 ){
        /* Do further verfication that this is explain output.  Abort if
        ** it is not */
        static const char *explainCols[] = {
           "addr", "opcode", "p1", "p2", "p3", "p4", "p5", "comment" };
        int jj;
        for(jj=0; jj<ArraySize(explainCols); jj++){
          if( strcmp(sqlite3_column_name(pSql,jj),explainCols[jj])!=0 ){
            p->cMode = p->mode;
            sqlite3_reset(pSql);
            return;
          }
        }
      }
      nAlloc += 100;
      p->aiIndent = (int*)sqlite3_realloc64(p->aiIndent, nAlloc*sizeof(int));
      abYield = (int*)sqlite3_realloc64(abYield, nAlloc*sizeof(int));
    }
    abYield[iOp] = str_in_array(zOp, azYield);
    p->aiIndent[iOp] = 0;
    p->nIndent = iOp+1;

    if( str_in_array(zOp, azNext) ){
      for(i=p2op; i<iOp; i++) p->aiIndent[i] += 2;
    }
    if( str_in_array(zOp, azGoto) && p2op<p->nIndent
     && (abYield[p2op] || sqlite3_column_int(pSql, 2))
    ){
      for(i=p2op; i<iOp; i++) p->aiIndent[i] += 2;
    }
  }

  p->iIndent = 0;
  sqlite3_free(abYield);
  sqlite3_reset(pSql);
}

/*
** Free the array allocated by explain_data_prepare().
*/
static void explain_data_delete(ShellState *p){
  sqlite3_free(p->aiIndent);
  p->aiIndent = 0;
  p->nIndent = 0;
  p->iIndent = 0;
}

/*
** Disable and restore .wheretrace and .selecttrace settings.
*/
#if defined(SQLITE_DEBUG) && defined(SQLITE_ENABLE_SELECTTRACE)
extern int sqlite3SelectTrace;
static int savedSelectTrace;
#endif
#if defined(SQLITE_DEBUG) && defined(SQLITE_ENABLE_WHERETRACE)
extern int sqlite3WhereTrace;
static int savedWhereTrace;
#endif
static void disable_debug_trace_modes(void){
#if defined(SQLITE_DEBUG) && defined(SQLITE_ENABLE_SELECTTRACE)
  savedSelectTrace = sqlite3SelectTrace;
  sqlite3SelectTrace = 0;
#endif
#if defined(SQLITE_DEBUG) && defined(SQLITE_ENABLE_WHERETRACE)
  savedWhereTrace = sqlite3WhereTrace;
  sqlite3WhereTrace = 0;
#endif
}
static void restore_debug_trace_modes(void){
#if defined(SQLITE_DEBUG) && defined(SQLITE_ENABLE_SELECTTRACE)
  sqlite3SelectTrace = savedSelectTrace;
#endif
#if defined(SQLITE_DEBUG) && defined(SQLITE_ENABLE_WHERETRACE)
  sqlite3WhereTrace = savedWhereTrace;
#endif
}

/*
** Run a prepared statement
*/
static void exec_prepared_stmt(
  ShellState *pArg,                                /* Pointer to ShellState */
  sqlite3_stmt *pStmt,                             /* Statment to run */
  int (*xCallback)(void*,int,char**,char**,int*)   /* Callback function */
){
  int rc;

  /* perform the first step.  this will tell us if we
  ** have a result set or not and how wide it is.
  */
  rc = sqlite3_step(pStmt);
  /* if we have a result set... */
  if( SQLITE_ROW == rc ){
    /* if we have a callback... */
    if( xCallback ){
      /* allocate space for col name ptr, value ptr, and type */
      int nCol = sqlite3_column_count(pStmt);
      void *pData = sqlite3_malloc64(3*nCol*sizeof(const char*) + 1);
      if( !pData ){
        rc = SQLITE_NOMEM;
      }else{
        char **azCols = (char **)pData;      /* Names of result columns */
        char **azVals = &azCols[nCol];       /* Results */
        int *aiTypes = (int *)&azVals[nCol]; /* Result types */
        int i, x;
        assert(sizeof(int) <= sizeof(char *));
        /* save off ptrs to column names */
        for(i=0; i<nCol; i++){
          azCols[i] = (char *)sqlite3_column_name(pStmt, i);
        }
        do{
          /* extract the data and data types */
          for(i=0; i<nCol; i++){
            aiTypes[i] = x = sqlite3_column_type(pStmt, i);
            if( x==SQLITE_BLOB && pArg && pArg->cMode==MODE_Insert ){
              azVals[i] = "";
            }else{
              azVals[i] = (char*)sqlite3_column_text(pStmt, i);
            }
            if( !azVals[i] && (aiTypes[i]!=SQLITE_NULL) ){
              rc = SQLITE_NOMEM;
              break; /* from for */
            }
          } /* end for */

          /* if data and types extracted successfully... */
          if( SQLITE_ROW == rc ){
            /* call the supplied callback with the result row data */
            if( xCallback(pArg, nCol, azVals, azCols, aiTypes) ){
              rc = SQLITE_ABORT;
            }else{
              rc = sqlite3_step(pStmt);
            }
          }
        } while( SQLITE_ROW == rc );
        sqlite3_free(pData);
      }
    }else{
      do{
        rc = sqlite3_step(pStmt);
      } while( rc == SQLITE_ROW );
    }
  }
}

/*
** Execute a statement or set of statements.  Print
** any result rows/columns depending on the current mode
** set via the supplied callback.
**
** This is very similar to SQLite's built-in sqlite3_exec()
** function except it takes a slightly different callback
** and callback data argument.
*/
static int shell_exec(
  sqlite3 *db,                              /* An open database */
  const char *zSql,                         /* SQL to be evaluated */
  int (*xCallback)(void*,int,char**,char**,int*),   /* Callback function */
                                            /* (not the same as sqlite3_exec) */
  ShellState *pArg,                         /* Pointer to ShellState */
  char **pzErrMsg                           /* Error msg written here */
){
  sqlite3_stmt *pStmt = NULL;     /* Statement to execute. */
  int rc = SQLITE_OK;             /* Return Code */
  int rc2;
  const char *zLeftover;          /* Tail of unprocessed SQL */

  if( pzErrMsg ){
    *pzErrMsg = NULL;
  }

  while( zSql[0] && (SQLITE_OK == rc) ){
    static const char *zStmtSql;
    rc = sqlite3_prepare_v2(db, zSql, -1, &pStmt, &zLeftover);
    if( SQLITE_OK != rc ){
      if( pzErrMsg ){
        *pzErrMsg = save_err_msg(db);
      }
    }else{
      if( !pStmt ){
        /* this happens for a comment or white-space */
        zSql = zLeftover;
        while( IsSpace(zSql[0]) ) zSql++;
        continue;
      }
      zStmtSql = sqlite3_sql(pStmt);
      if( zStmtSql==0 ) zStmtSql = "";
      while( IsSpace(zStmtSql[0]) ) zStmtSql++;

      /* save off the prepared statment handle and reset row count */
      if( pArg ){
        pArg->pStmt = pStmt;
        pArg->cnt = 0;
      }

      /* echo the sql statement if echo on */
      if( pArg && ShellHasFlag(pArg, SHFLG_Echo) ){
        utf8_printf(pArg->out, "%s\n", zStmtSql ? zStmtSql : zSql);
      }

      /* Show the EXPLAIN QUERY PLAN if .eqp is on */
      if( pArg && pArg->autoEQP && sqlite3_strlike("EXPLAIN%",zStmtSql,0)!=0 ){
        sqlite3_stmt *pExplain;
        char *zEQP;
        disable_debug_trace_modes();
        zEQP = sqlite3_mprintf("EXPLAIN QUERY PLAN %s", zStmtSql);
        rc = sqlite3_prepare_v2(db, zEQP, -1, &pExplain, 0);
        if( rc==SQLITE_OK ){
          while( sqlite3_step(pExplain)==SQLITE_ROW ){
            raw_printf(pArg->out,"--EQP-- %d,",sqlite3_column_int(pExplain, 0));
            raw_printf(pArg->out,"%d,", sqlite3_column_int(pExplain, 1));
            raw_printf(pArg->out,"%d,", sqlite3_column_int(pExplain, 2));
            utf8_printf(pArg->out,"%s\n", sqlite3_column_text(pExplain, 3));
          }
        }
        sqlite3_finalize(pExplain);
        sqlite3_free(zEQP);
        if( pArg->autoEQP>=2 ){
          /* Also do an EXPLAIN for ".eqp full" mode */
          zEQP = sqlite3_mprintf("EXPLAIN %s", zStmtSql);
          rc = sqlite3_prepare_v2(db, zEQP, -1, &pExplain, 0);
          if( rc==SQLITE_OK ){
            pArg->cMode = MODE_Explain;
            explain_data_prepare(pArg, pExplain);
            exec_prepared_stmt(pArg, pExplain, xCallback);
            explain_data_delete(pArg);
          }
          sqlite3_finalize(pExplain);
          sqlite3_free(zEQP);
        }
        restore_debug_trace_modes();
      }

      if( pArg ){
        pArg->cMode = pArg->mode;
        if( pArg->autoExplain
         && sqlite3_column_count(pStmt)==8
         && sqlite3_strlike("EXPLAIN%", zStmtSql,0)==0
        ){
          pArg->cMode = MODE_Explain;
        }

        /* If the shell is currently in ".explain" mode, gather the extra
        ** data required to add indents to the output.*/
        if( pArg->cMode==MODE_Explain ){
          explain_data_prepare(pArg, pStmt);
        }
      }

      exec_prepared_stmt(pArg, pStmt, xCallback);
      explain_data_delete(pArg);

      /* print usage stats if stats on */
      if( pArg && pArg->statsOn ){
        display_stats(db, pArg, 0);
      }

      /* print loop-counters if required */
      if( pArg && pArg->scanstatsOn ){
        display_scanstats(db, pArg);
      }

      /* Finalize the statement just executed. If this fails, save a
      ** copy of the error message. Otherwise, set zSql to point to the
      ** next statement to execute. */
      rc2 = sqlite3_finalize(pStmt);
      if( rc!=SQLITE_NOMEM ) rc = rc2;
      if( rc==SQLITE_OK ){
        zSql = zLeftover;
        while( IsSpace(zSql[0]) ) zSql++;
      }else if( pzErrMsg ){
        *pzErrMsg = save_err_msg(db);
      }

      /* clear saved stmt handle */
      if( pArg ){
        pArg->pStmt = NULL;
      }
    }
  } /* end while */

  return rc;
}

/*
** Release memory previously allocated by tableColumnList().
*/
static void freeColumnList(char **azCol){
  int i;
  for(i=1; azCol[i]; i++){
    sqlite3_free(azCol[i]);
  }
  /* azCol[0] is a static string */
  sqlite3_free(azCol);
}

/*
** Return a list of pointers to strings which are the names of all
** columns in table zTab.   The memory to hold the names is dynamically
** allocated and must be released by the caller using a subsequent call
** to freeColumnList().
**
** The azCol[0] entry is usually NULL.  However, if zTab contains a rowid
** value that needs to be preserved, then azCol[0] is filled in with the
** name of the rowid column.
**
** The first regular column in the table is azCol[1].  The list is terminated
** by an entry with azCol[i]==0.
*/
static char **tableColumnList(ShellState *p, const char *zTab){
  char **azCol = 0;
  sqlite3_stmt *pStmt;
  char *zSql;
  int nCol = 0;
  int nAlloc = 0;
  int nPK = 0;       /* Number of PRIMARY KEY columns seen */
  int isIPK = 0;     /* True if one PRIMARY KEY column of type INTEGER */
  int preserveRowid = ShellHasFlag(p, SHFLG_PreserveRowid);
  int rc;

  zSql = sqlite3_mprintf("PRAGMA table_info=%Q", zTab);
  rc = sqlite3_prepare_v2(p->db, zSql, -1, &pStmt, 0);
  sqlite3_free(zSql);
  if( rc ) return 0;
  while( sqlite3_step(pStmt)==SQLITE_ROW ){
    if( nCol>=nAlloc-2 ){
      nAlloc = nAlloc*2 + nCol + 10;
      azCol = sqlite3_realloc(azCol, nAlloc*sizeof(azCol[0]));
      if( azCol==0 ){
        raw_printf(stderr, "Error: out of memory\n");
        exit(1);
      }
    }
    azCol[++nCol] = sqlite3_mprintf("%s", sqlite3_column_text(pStmt, 1));
    if( sqlite3_column_int(pStmt, 5) ){
      nPK++;
      if( nPK==1
       && sqlite3_stricmp((const char*)sqlite3_column_text(pStmt,2),
                          "INTEGER")==0 
      ){
        isIPK = 1;
      }else{
        isIPK = 0;
      }
    }
  }
  sqlite3_finalize(pStmt);
  azCol[0] = 0;
  azCol[nCol+1] = 0;

  /* The decision of whether or not a rowid really needs to be preserved
  ** is tricky.  We never need to preserve a rowid for a WITHOUT ROWID table
  ** or a table with an INTEGER PRIMARY KEY.  We are unable to preserve
  ** rowids on tables where the rowid is inaccessible because there are other
  ** columns in the table named "rowid", "_rowid_", and "oid".
  */
  if( preserveRowid && isIPK ){
    /* If a single PRIMARY KEY column with type INTEGER was seen, then it
    ** might be an alise for the ROWID.  But it might also be a WITHOUT ROWID
    ** table or a INTEGER PRIMARY KEY DESC column, neither of which are
    ** ROWID aliases.  To distinguish these cases, check to see if
    ** there is a "pk" entry in "PRAGMA index_list".  There will be
    ** no "pk" index if the PRIMARY KEY really is an alias for the ROWID.
    */
    zSql = sqlite3_mprintf("SELECT 1 FROM pragma_index_list(%Q)"
                           " WHERE origin='pk'", zTab);
    rc = sqlite3_prepare_v2(p->db, zSql, -1, &pStmt, 0);
    sqlite3_free(zSql);
    if( rc ){
      freeColumnList(azCol);
      return 0;
    }
    rc = sqlite3_step(pStmt);
    sqlite3_finalize(pStmt);
    preserveRowid = rc==SQLITE_ROW;
  }
  if( preserveRowid ){
    /* Only preserve the rowid if we can find a name to use for the
    ** rowid */
    static char *azRowid[] = { "rowid", "_rowid_", "oid" };
    int i, j;
    for(j=0; j<3; j++){
      for(i=1; i<=nCol; i++){
        if( sqlite3_stricmp(azRowid[j],azCol[i])==0 ) break;
      }
      if( i>nCol ){
        /* At this point, we know that azRowid[j] is not the name of any
        ** ordinary column in the table.  Verify that azRowid[j] is a valid
        ** name for the rowid before adding it to azCol[0].  WITHOUT ROWID
        ** tables will fail this last check */
        rc = sqlite3_table_column_metadata(p->db,0,zTab,azRowid[j],0,0,0,0,0);
        if( rc==SQLITE_OK ) azCol[0] = azRowid[j];
        break;
      }
    }
  }
  return azCol;
}

/*
** Toggle the reverse_unordered_selects setting.
*/
static void toggleSelectOrder(sqlite3 *db){
  sqlite3_stmt *pStmt = 0;
  int iSetting = 0;
  char zStmt[100];
  sqlite3_prepare_v2(db, "PRAGMA reverse_unordered_selects", -1, &pStmt, 0);
  if( sqlite3_step(pStmt)==SQLITE_ROW ){
    iSetting = sqlite3_column_int(pStmt, 0);
  }
  sqlite3_finalize(pStmt);
  sqlite3_snprintf(sizeof(zStmt), zStmt,
       "PRAGMA reverse_unordered_selects(%d)", !iSetting);
  sqlite3_exec(db, zStmt, 0, 0, 0);
}

/*
** This is a different callback routine used for dumping the database.
** Each row received by this callback consists of a table name,
** the table type ("index" or "table") and SQL to create the table.
** This routine should print text sufficient to recreate the table.
*/
static int dump_callback(void *pArg, int nArg, char **azArg, char **azNotUsed){
  int rc;
  const char *zTable;
  const char *zType;
  const char *zSql;
  ShellState *p = (ShellState *)pArg;

  UNUSED_PARAMETER(azNotUsed);
  if( nArg!=3 ) return 1;
  zTable = azArg[0];
  zType = azArg[1];
  zSql = azArg[2];

  if( strcmp(zTable, "sqlite_sequence")==0 ){
    raw_printf(p->out, "DELETE FROM sqlite_sequence;\n");
  }else if( sqlite3_strglob("sqlite_stat?", zTable)==0 ){
    raw_printf(p->out, "ANALYZE sqlite_master;\n");
  }else if( strncmp(zTable, "sqlite_", 7)==0 ){
    return 0;
  }else if( strncmp(zSql, "CREATE VIRTUAL TABLE", 20)==0 ){
    char *zIns;
    if( !p->writableSchema ){
      raw_printf(p->out, "PRAGMA writable_schema=ON;\n");
      p->writableSchema = 1;
    }
    zIns = sqlite3_mprintf(
       "INSERT INTO sqlite_master(type,name,tbl_name,rootpage,sql)"
       "VALUES('table','%q','%q',0,'%q');",
       zTable, zTable, zSql);
    utf8_printf(p->out, "%s\n", zIns);
    sqlite3_free(zIns);
    return 0;
  }else{
    printSchemaLine(p->out, zSql, ";\n");
  }

  if( strcmp(zType, "table")==0 ){
    ShellText sSelect;
    ShellText sTable;
    char **azCol;
    int i;
    char *savedDestTable;
    int savedMode;

    azCol = tableColumnList(p, zTable);
    if( azCol==0 ){
      p->nErr++;
      return 0;
    }

    /* Always quote the table name, even if it appears to be pure ascii,
    ** in case it is a keyword. Ex:  INSERT INTO "table" ... */
    initText(&sTable);
    appendText(&sTable, zTable, quoteChar(zTable));
    /* If preserving the rowid, add a column list after the table name.
    ** In other words:  "INSERT INTO tab(rowid,a,b,c,...) VALUES(...)"
    ** instead of the usual "INSERT INTO tab VALUES(...)".
    */
    if( azCol[0] ){
      appendText(&sTable, "(", 0);
      appendText(&sTable, azCol[0], 0);
      for(i=1; azCol[i]; i++){
        appendText(&sTable, ",", 0);
        appendText(&sTable, azCol[i], quoteChar(azCol[i]));
      }
      appendText(&sTable, ")", 0);
    }

    /* Build an appropriate SELECT statement */
    initText(&sSelect);
    appendText(&sSelect, "SELECT ", 0);
    if( azCol[0] ){
      appendText(&sSelect, azCol[0], 0);
      appendText(&sSelect, ",", 0);
    }
    for(i=1; azCol[i]; i++){
      appendText(&sSelect, azCol[i], quoteChar(azCol[i]));
      if( azCol[i+1] ){
        appendText(&sSelect, ",", 0);
      }
    }
    freeColumnList(azCol);
    appendText(&sSelect, " FROM ", 0);
    appendText(&sSelect, zTable, quoteChar(zTable));

    savedDestTable = p->zDestTable;
    savedMode = p->mode;
    p->zDestTable = sTable.z;
    p->mode = p->cMode = MODE_Insert;
    rc = shell_exec(p->db, sSelect.z, shell_callback, p, 0);
    if( (rc&0xff)==SQLITE_CORRUPT ){
      raw_printf(p->out, "/****** CORRUPTION ERROR *******/\n");
      toggleSelectOrder(p->db);
      shell_exec(p->db, sSelect.z, shell_callback, p, 0);
      toggleSelectOrder(p->db);
    }
    p->zDestTable = savedDestTable;
    p->mode = savedMode;
    freeText(&sTable);
    freeText(&sSelect);
    if( rc ) p->nErr++;
  }
  return 0;
}

/*
** Run zQuery.  Use dump_callback() as the callback routine so that
** the contents of the query are output as SQL statements.
**
** If we get a SQLITE_CORRUPT error, rerun the query after appending
** "ORDER BY rowid DESC" to the end.
*/
static int run_schema_dump_query(
  ShellState *p,
  const char *zQuery
){
  int rc;
  char *zErr = 0;
  rc = sqlite3_exec(p->db, zQuery, dump_callback, p, &zErr);
  if( rc==SQLITE_CORRUPT ){
    char *zQ2;
    int len = strlen30(zQuery);
    raw_printf(p->out, "/****** CORRUPTION ERROR *******/\n");
    if( zErr ){
      utf8_printf(p->out, "/****** %s ******/\n", zErr);
      sqlite3_free(zErr);
      zErr = 0;
    }
    zQ2 = malloc( len+100 );
    if( zQ2==0 ) return rc;
    sqlite3_snprintf(len+100, zQ2, "%s ORDER BY rowid DESC", zQuery);
    rc = sqlite3_exec(p->db, zQ2, dump_callback, p, &zErr);
    if( rc ){
      utf8_printf(p->out, "/****** ERROR: %s ******/\n", zErr);
    }else{
      rc = SQLITE_CORRUPT;
    }
    sqlite3_free(zErr);
    free(zQ2);
  }
  return rc;
}

/*
** Text of a help message
*/
static char zHelp[] =
#ifndef SQLITE_OMIT_AUTHORIZATION
  ".auth ON|OFF           Show authorizer callbacks\n"
#endif
  ".backup ?DB? FILE      Backup DB (default \"main\") to FILE\n"
  ".bail on|off           Stop after hitting an error.  Default OFF\n"
  ".binary on|off         Turn binary output on or off.  Default OFF\n"
  ".changes on|off        Show number of rows changed by SQL\n"
  ".check GLOB            Fail if output since .testcase does not match\n"
  ".clone NEWDB           Clone data into NEWDB from the existing database\n"
  ".databases             List names and files of attached databases\n"
  ".dbinfo ?DB?           Show status information about the database\n"
  ".dump ?TABLE? ...      Dump the database in an SQL text format\n"
  "                         If TABLE specified, only dump tables matching\n"
  "                         LIKE pattern TABLE.\n"
  ".echo on|off           Turn command echo on or off\n"
  ".eqp on|off|full       Enable or disable automatic EXPLAIN QUERY PLAN\n"
  ".exit                  Exit this program\n"
  ".explain ?on|off|auto? Turn EXPLAIN output mode on or off or to automatic\n"
  ".fullschema ?--indent? Show schema and the content of sqlite_stat tables\n"
  ".headers on|off        Turn display of headers on or off\n"
  ".help                  Show this message\n"
  ".import FILE TABLE     Import data from FILE into TABLE\n"
#ifndef SQLITE_OMIT_TEST_CONTROL
  ".imposter INDEX TABLE  Create imposter table TABLE on index INDEX\n"
#endif
  ".indexes ?TABLE?       Show names of all indexes\n"
  "                         If TABLE specified, only show indexes for tables\n"
  "                         matching LIKE pattern TABLE.\n"
#ifdef SQLITE_ENABLE_IOTRACE
  ".iotrace FILE          Enable I/O diagnostic logging to FILE\n"
#endif
  ".limit ?LIMIT? ?VAL?   Display or change the value of an SQLITE_LIMIT\n"
  ".lint OPTIONS          Report potential schema issues. Options:\n"
  "                         fkey-indexes     Find missing foreign key indexes\n"
#ifndef SQLITE_OMIT_LOAD_EXTENSION
  ".load FILE ?ENTRY?     Load an extension library\n"
#endif
  ".log FILE|off          Turn logging on or off.  FILE can be stderr/stdout\n"
  ".mode MODE ?TABLE?     Set output mode where MODE is one of:\n"
  "                         ascii    Columns/rows delimited by 0x1F and 0x1E\n"
  "                         csv      Comma-separated values\n"
  "                         column   Left-aligned columns.  (See .width)\n"
  "                         html     HTML <table> code\n"
  "                         insert   SQL insert statements for TABLE\n"
  "                         line     One value per line\n"
  "                         list     Values delimited by \"|\"\n"
  "                         quote    Escape answers as for SQL\n"
  "                         tabs     Tab-separated values\n"
  "                         tcl      TCL list elements\n"
  ".nullvalue STRING      Use STRING in place of NULL values\n"
  ".once FILENAME         Output for the next SQL command only to FILENAME\n"
  ".open ?--new? ?FILE?   Close existing database and reopen FILE\n"
  "                         The --new starts with an empty file\n"
  ".output ?FILENAME?     Send output to FILENAME or stdout\n"
  ".print STRING...       Print literal STRING\n"
  ".prompt MAIN CONTINUE  Replace the standard prompts\n"
  ".quit                  Exit this program\n"
  ".read FILENAME         Execute SQL in FILENAME\n"
  ".restore ?DB? FILE     Restore content of DB (default \"main\") from FILE\n"
  ".save FILE             Write in-memory database into FILE\n"
  ".scanstats on|off      Turn sqlite3_stmt_scanstatus() metrics on or off\n"
  ".schema ?PATTERN?      Show the CREATE statements matching PATTERN\n"
  "                          Add --indent for pretty-printing\n"
  ".selftest ?--init?     Run tests defined in the SELFTEST table\n"
  ".separator COL ?ROW?   Change the column separator and optionally the row\n"
  "                         separator for both the output mode and .import\n"
#if defined(SQLITE_ENABLE_SESSION)
  ".session CMD ...       Create or control sessions\n"
#endif
  ".sha3sum ?OPTIONS...?  Compute a SHA3 hash of database content\n"
  ".shell CMD ARGS...     Run CMD ARGS... in a system shell\n"
  ".show                  Show the current values for various settings\n"
  ".stats ?on|off?        Show stats or turn stats on or off\n"
  ".system CMD ARGS...    Run CMD ARGS... in a system shell\n"
  ".tables ?TABLE?        List names of tables\n"
  "                         If TABLE specified, only list tables matching\n"
  "                         LIKE pattern TABLE.\n"
  ".testcase NAME         Begin redirecting output to 'testcase-out.txt'\n"
  ".timeout MS            Try opening locked tables for MS milliseconds\n"
  ".timer on|off          Turn SQL timer on or off\n"
  ".trace FILE|off        Output each SQL statement as it is run\n"
  ".vfsinfo ?AUX?         Information about the top-level VFS\n"
  ".vfslist               List all available VFSes\n"
  ".vfsname ?AUX?         Print the name of the VFS stack\n"
  ".width NUM1 NUM2 ...   Set column widths for \"column\" mode\n"
  "                         Negative values right-justify\n"
;

#if defined(SQLITE_ENABLE_SESSION)
/*
** Print help information for the ".sessions" command
*/
void session_help(ShellState *p){
  raw_printf(p->out,
    ".session ?NAME? SUBCOMMAND ?ARGS...?\n"
    "If ?NAME? is omitted, the first defined session is used.\n"
    "Subcommands:\n"
    "   attach TABLE             Attach TABLE\n"
    "   changeset FILE           Write a changeset into FILE\n"
    "   close                    Close one session\n"
    "   enable ?BOOLEAN?         Set or query the enable bit\n"
    "   filter GLOB...           Reject tables matching GLOBs\n"
    "   indirect ?BOOLEAN?       Mark or query the indirect status\n"
    "   isempty                  Query whether the session is empty\n"
    "   list                     List currently open session names\n"
    "   open DB NAME             Open a new session on DB\n"
    "   patchset FILE            Write a patchset into FILE\n"
  );
}
#endif


/* Forward reference */
static int process_input(ShellState *p, FILE *in);

/*
** Read the content of file zName into memory obtained from sqlite3_malloc64()
** and return a pointer to the buffer. The caller is responsible for freeing 
** the memory. 
**
** If parameter pnByte is not NULL, (*pnByte) is set to the number of bytes
** read.
**
** For convenience, a nul-terminator byte is always appended to the data read
** from the file before the buffer is returned. This byte is not included in
** the final value of (*pnByte), if applicable.
**
** NULL is returned if any error is encountered. The final value of *pnByte
** is undefined in this case.
*/
static char *readFile(const char *zName, int *pnByte){
  FILE *in = fopen(zName, "rb");
  long nIn;
  size_t nRead;
  char *pBuf;
  if( in==0 ) return 0;
  fseek(in, 0, SEEK_END);
  nIn = ftell(in);
  rewind(in);
  pBuf = sqlite3_malloc64( nIn+1 );
  if( pBuf==0 ) return 0;
  nRead = fread(pBuf, nIn, 1, in);
  fclose(in);
  if( nRead!=1 ){
    sqlite3_free(pBuf);
    return 0;
  }
  pBuf[nIn] = 0;
  if( pnByte ) *pnByte = nIn;
  return pBuf;
}

/*
** Implementation of the "readfile(X)" SQL function.  The entire content
** of the file named X is read and returned as a BLOB.  NULL is returned
** if the file does not exist or is unreadable.
*/
static void readfileFunc(
  sqlite3_context *context,
  int argc,
  sqlite3_value **argv
){
  const char *zName;
  void *pBuf;
  int nBuf;

  UNUSED_PARAMETER(argc);
  zName = (const char*)sqlite3_value_text(argv[0]);
  if( zName==0 ) return;
  pBuf = readFile(zName, &nBuf);
  if( pBuf ) sqlite3_result_blob(context, pBuf, nBuf, sqlite3_free);
}

/*
** Implementation of the "writefile(X,Y)" SQL function.  The argument Y
** is written into file X.  The number of bytes written is returned.  Or
** NULL is returned if something goes wrong, such as being unable to open
** file X for writing.
*/
static void writefileFunc(
  sqlite3_context *context,
  int argc,
  sqlite3_value **argv
){
  FILE *out;
  const char *z;
  sqlite3_int64 rc;
  const char *zFile;

  UNUSED_PARAMETER(argc);
  zFile = (const char*)sqlite3_value_text(argv[0]);
  if( zFile==0 ) return;
  out = fopen(zFile, "wb");
  if( out==0 ) return;
  z = (const char*)sqlite3_value_blob(argv[1]);
  if( z==0 ){
    rc = 0;
  }else{
    rc = fwrite(z, 1, sqlite3_value_bytes(argv[1]), out);
  }
  fclose(out);
  sqlite3_result_int64(context, rc);
}

#if defined(SQLITE_ENABLE_SESSION)
/*
** Close a single OpenSession object and release all of its associated
** resources.
*/
static void session_close(OpenSession *pSession){
  int i;
  sqlite3session_delete(pSession->p);
  sqlite3_free(pSession->zName);
  for(i=0; i<pSession->nFilter; i++){
    sqlite3_free(pSession->azFilter[i]);
  }
  sqlite3_free(pSession->azFilter);
  memset(pSession, 0, sizeof(OpenSession));
}
#endif

/*
** Close all OpenSession objects and release all associated resources.
*/
#if defined(SQLITE_ENABLE_SESSION)
static void session_close_all(ShellState *p){
  int i;
  for(i=0; i<p->nSession; i++){
    session_close(&p->aSession[i]);
  }
  p->nSession = 0;
}
#else
# define session_close_all(X)
#endif

/*
** Implementation of the xFilter function for an open session.  Omit
** any tables named by ".session filter" but let all other table through.
*/
#if defined(SQLITE_ENABLE_SESSION)
static int session_filter(void *pCtx, const char *zTab){
  OpenSession *pSession = (OpenSession*)pCtx;
  int i;
  for(i=0; i<pSession->nFilter; i++){
    if( sqlite3_strglob(pSession->azFilter[i], zTab)==0 ) return 0;
  }
  return 1;
}
#endif

/*
** Make sure the database is open.  If it is not, then open it.  If
** the database fails to open, print an error message and exit.
*/
static void open_db(ShellState *p, int keepAlive){
  if( p->db==0 ){
    sqlite3_initialize();
    sqlite3_open(p->zDbFilename, &p->db);
    globalDb = p->db;
    if( p->db==0 || SQLITE_OK!=sqlite3_errcode(p->db) ){
      utf8_printf(stderr,"Error: unable to open database \"%s\": %s\n",
          p->zDbFilename, sqlite3_errmsg(p->db));
      if( keepAlive ) return;
      exit(1);
    }
#ifndef SQLITE_OMIT_LOAD_EXTENSION
    sqlite3_enable_load_extension(p->db, 1);
#endif
    sqlite3_create_function(p->db, "readfile", 1, SQLITE_UTF8, 0,
                            readfileFunc, 0, 0);
    sqlite3_create_function(p->db, "writefile", 2, SQLITE_UTF8, 0,
                            writefileFunc, 0, 0);
    sqlite3_create_function(p->db, "sha3", 1, SQLITE_UTF8, 0,
                            sha3Func, 0, 0);
    sqlite3_create_function(p->db, "sha3", 2, SQLITE_UTF8, 0,
                            sha3Func, 0, 0);
    sqlite3_create_function(p->db, "sha3_query", 1, SQLITE_UTF8, 0,
                            sha3QueryFunc, 0, 0);
    sqlite3_create_function(p->db, "sha3_query", 2, SQLITE_UTF8, 0,
                            sha3QueryFunc, 0, 0);
  }
}

/*
** Do C-language style dequoting.
**
**    \a    -> alarm
**    \b    -> backspace
**    \t    -> tab
**    \n    -> newline
**    \v    -> vertical tab
**    \f    -> form feed
**    \r    -> carriage return
**    \s    -> space
**    \"    -> "
**    \'    -> '
**    \\    -> backslash
**    \NNN  -> ascii character NNN in octal
*/
static void resolve_backslashes(char *z){
  int i, j;
  char c;
  while( *z && *z!='\\' ) z++;
  for(i=j=0; (c = z[i])!=0; i++, j++){
    if( c=='\\' && z[i+1]!=0 ){
      c = z[++i];
      if( c=='a' ){
        c = '\a';
      }else if( c=='b' ){
        c = '\b';
      }else if( c=='t' ){
        c = '\t';
      }else if( c=='n' ){
        c = '\n';
      }else if( c=='v' ){
        c = '\v';
      }else if( c=='f' ){
        c = '\f';
      }else if( c=='r' ){
        c = '\r';
      }else if( c=='"' ){
        c = '"';
      }else if( c=='\'' ){
        c = '\'';
      }else if( c=='\\' ){
        c = '\\';
      }else if( c>='0' && c<='7' ){
        c -= '0';
        if( z[i+1]>='0' && z[i+1]<='7' ){
          i++;
          c = (c<<3) + z[i] - '0';
          if( z[i+1]>='0' && z[i+1]<='7' ){
            i++;
            c = (c<<3) + z[i] - '0';
          }
        }
      }
    }
    z[j] = c;
  }
  if( j<i ) z[j] = 0;
}

/*
** Return the value of a hexadecimal digit.  Return -1 if the input
** is not a hex digit.
*/
static int hexDigitValue(char c){
  if( c>='0' && c<='9' ) return c - '0';
  if( c>='a' && c<='f' ) return c - 'a' + 10;
  if( c>='A' && c<='F' ) return c - 'A' + 10;
  return -1;
}

/*
** Interpret zArg as an integer value, possibly with suffixes.
*/
static sqlite3_int64 integerValue(const char *zArg){
  sqlite3_int64 v = 0;
  static const struct { char *zSuffix; int iMult; } aMult[] = {
    { "KiB", 1024 },
    { "MiB", 1024*1024 },
    { "GiB", 1024*1024*1024 },
    { "KB",  1000 },
    { "MB",  1000000 },
    { "GB",  1000000000 },
    { "K",   1000 },
    { "M",   1000000 },
    { "G",   1000000000 },
  };
  int i;
  int isNeg = 0;
  if( zArg[0]=='-' ){
    isNeg = 1;
    zArg++;
  }else if( zArg[0]=='+' ){
    zArg++;
  }
  if( zArg[0]=='0' && zArg[1]=='x' ){
    int x;
    zArg += 2;
    while( (x = hexDigitValue(zArg[0]))>=0 ){
      v = (v<<4) + x;
      zArg++;
    }
  }else{
    while( IsDigit(zArg[0]) ){
      v = v*10 + zArg[0] - '0';
      zArg++;
    }
  }
  for(i=0; i<ArraySize(aMult); i++){
    if( sqlite3_stricmp(aMult[i].zSuffix, zArg)==0 ){
      v *= aMult[i].iMult;
      break;
    }
  }
  return isNeg? -v : v;
}

/*
** Interpret zArg as either an integer or a boolean value.  Return 1 or 0
** for TRUE and FALSE.  Return the integer value if appropriate.
*/
static int booleanValue(const char *zArg){
  int i;
  if( zArg[0]=='0' && zArg[1]=='x' ){
    for(i=2; hexDigitValue(zArg[i])>=0; i++){}
  }else{
    for(i=0; zArg[i]>='0' && zArg[i]<='9'; i++){}
  }
  if( i>0 && zArg[i]==0 ) return (int)(integerValue(zArg) & 0xffffffff);
  if( sqlite3_stricmp(zArg, "on")==0 || sqlite3_stricmp(zArg,"yes")==0 ){
    return 1;
  }
  if( sqlite3_stricmp(zArg, "off")==0 || sqlite3_stricmp(zArg,"no")==0 ){
    return 0;
  }
  utf8_printf(stderr, "ERROR: Not a boolean value: \"%s\". Assuming \"no\".\n",
          zArg);
  return 0;
}

/*
** Set or clear a shell flag according to a boolean value.
*/
static void setOrClearFlag(ShellState *p, unsigned mFlag, const char *zArg){
  if( booleanValue(zArg) ){
    ShellSetFlag(p, mFlag);
  }else{
    ShellClearFlag(p, mFlag);
  }
}

/*
** Close an output file, assuming it is not stderr or stdout
*/
static void output_file_close(FILE *f){
  if( f && f!=stdout && f!=stderr ) fclose(f);
}

/*
** Try to open an output file.   The names "stdout" and "stderr" are
** recognized and do the right thing.  NULL is returned if the output
** filename is "off".
*/
static FILE *output_file_open(const char *zFile){
  FILE *f;
  if( strcmp(zFile,"stdout")==0 ){
    f = stdout;
  }else if( strcmp(zFile, "stderr")==0 ){
    f = stderr;
  }else if( strcmp(zFile, "off")==0 ){
    f = 0;
  }else{
    f = fopen(zFile, "wb");
    if( f==0 ){
      utf8_printf(stderr, "Error: cannot open \"%s\"\n", zFile);
    }
  }
  return f;
}

#if !defined(SQLITE_UNTESTABLE)
#if !defined(SQLITE_OMIT_TRACE) && !defined(SQLITE_OMIT_FLOATING_POINT)
/*
** A routine for handling output from sqlite3_trace().
*/
static int sql_trace_callback(
  unsigned mType,
  void *pArg,
  void *pP,
  void *pX
){
  FILE *f = (FILE*)pArg;
  UNUSED_PARAMETER(mType);
  UNUSED_PARAMETER(pP);
  if( f ){
    const char *z = (const char*)pX;
    int i = (int)strlen(z);
    while( i>0 && z[i-1]==';' ){ i--; }
    utf8_printf(f, "%.*s;\n", i, z);
  }
  return 0;
}
#endif
#endif

/*
** A no-op routine that runs with the ".breakpoint" doc-command.  This is
** a useful spot to set a debugger breakpoint.
*/
static void test_breakpoint(void){
  static int nCall = 0;
  nCall++;
}

/*
** An object used to read a CSV and other files for import.
*/
typedef struct ImportCtx ImportCtx;
struct ImportCtx {
  const char *zFile;  /* Name of the input file */
  FILE *in;           /* Read the CSV text from this input stream */
  char *z;            /* Accumulated text for a field */
  int n;              /* Number of bytes in z */
  int nAlloc;         /* Space allocated for z[] */
  int nLine;          /* Current line number */
  int cTerm;          /* Character that terminated the most recent field */
  int cColSep;        /* The column separator character.  (Usually ",") */
  int cRowSep;        /* The row separator character.  (Usually "\n") */
};

/* Append a single byte to z[] */
static void import_append_char(ImportCtx *p, int c){
  if( p->n+1>=p->nAlloc ){
    p->nAlloc += p->nAlloc + 100;
    p->z = sqlite3_realloc64(p->z, p->nAlloc);
    if( p->z==0 ){
      raw_printf(stderr, "out of memory\n");
      exit(1);
    }
  }
  p->z[p->n++] = (char)c;
}

/* Read a single field of CSV text.  Compatible with rfc4180 and extended
** with the option of having a separator other than ",".
**
**   +  Input comes from p->in.
**   +  Store results in p->z of length p->n.  Space to hold p->z comes
**      from sqlite3_malloc64().
**   +  Use p->cSep as the column separator.  The default is ",".
**   +  Use p->rSep as the row separator.  The default is "\n".
**   +  Keep track of the line number in p->nLine.
**   +  Store the character that terminates the field in p->cTerm.  Store
**      EOF on end-of-file.
**   +  Report syntax errors on stderr
*/
static char *SQLITE_CDECL csv_read_one_field(ImportCtx *p){
  int c;
  int cSep = p->cColSep;
  int rSep = p->cRowSep;
  p->n = 0;
  c = fgetc(p->in);
  if( c==EOF || seenInterrupt ){
    p->cTerm = EOF;
    return 0;
  }
  if( c=='"' ){
    int pc, ppc;
    int startLine = p->nLine;
    int cQuote = c;
    pc = ppc = 0;
    while( 1 ){
      c = fgetc(p->in);
      if( c==rSep ) p->nLine++;
      if( c==cQuote ){
        if( pc==cQuote ){
          pc = 0;
          continue;
        }
      }
      if( (c==cSep && pc==cQuote)
       || (c==rSep && pc==cQuote)
       || (c==rSep && pc=='\r' && ppc==cQuote)
       || (c==EOF && pc==cQuote)
      ){
        do{ p->n--; }while( p->z[p->n]!=cQuote );
        p->cTerm = c;
        break;
      }
      if( pc==cQuote && c!='\r' ){
        utf8_printf(stderr, "%s:%d: unescaped %c character\n",
                p->zFile, p->nLine, cQuote);
      }
      if( c==EOF ){
        utf8_printf(stderr, "%s:%d: unterminated %c-quoted field\n",
                p->zFile, startLine, cQuote);
        p->cTerm = c;
        break;
      }
      import_append_char(p, c);
      ppc = pc;
      pc = c;
    }
  }else{
    while( c!=EOF && c!=cSep && c!=rSep ){
      import_append_char(p, c);
      c = fgetc(p->in);
    }
    if( c==rSep ){
      p->nLine++;
      if( p->n>0 && p->z[p->n-1]=='\r' ) p->n--;
    }
    p->cTerm = c;
  }
  if( p->z ) p->z[p->n] = 0;
  return p->z;
}

/* Read a single field of ASCII delimited text.
**
**   +  Input comes from p->in.
**   +  Store results in p->z of length p->n.  Space to hold p->z comes
**      from sqlite3_malloc64().
**   +  Use p->cSep as the column separator.  The default is "\x1F".
**   +  Use p->rSep as the row separator.  The default is "\x1E".
**   +  Keep track of the row number in p->nLine.
**   +  Store the character that terminates the field in p->cTerm.  Store
**      EOF on end-of-file.
**   +  Report syntax errors on stderr
*/
static char *SQLITE_CDECL ascii_read_one_field(ImportCtx *p){
  int c;
  int cSep = p->cColSep;
  int rSep = p->cRowSep;
  p->n = 0;
  c = fgetc(p->in);
  if( c==EOF || seenInterrupt ){
    p->cTerm = EOF;
    return 0;
  }
  while( c!=EOF && c!=cSep && c!=rSep ){
    import_append_char(p, c);
    c = fgetc(p->in);
  }
  if( c==rSep ){
    p->nLine++;
  }
  p->cTerm = c;
  if( p->z ) p->z[p->n] = 0;
  return p->z;
}

/*
** Try to transfer data for table zTable.  If an error is seen while
** moving forward, try to go backwards.  The backwards movement won't
** work for WITHOUT ROWID tables.
*/
static void tryToCloneData(
  ShellState *p,
  sqlite3 *newDb,
  const char *zTable
){
  sqlite3_stmt *pQuery = 0;
  sqlite3_stmt *pInsert = 0;
  char *zQuery = 0;
  char *zInsert = 0;
  int rc;
  int i, j, n;
  int nTable = (int)strlen(zTable);
  int k = 0;
  int cnt = 0;
  const int spinRate = 10000;

  zQuery = sqlite3_mprintf("SELECT * FROM \"%w\"", zTable);
  rc = sqlite3_prepare_v2(p->db, zQuery, -1, &pQuery, 0);
  if( rc ){
    utf8_printf(stderr, "Error %d: %s on [%s]\n",
            sqlite3_extended_errcode(p->db), sqlite3_errmsg(p->db),
            zQuery);
    goto end_data_xfer;
  }
  n = sqlite3_column_count(pQuery);
  zInsert = sqlite3_malloc64(200 + nTable + n*3);
  if( zInsert==0 ){
    raw_printf(stderr, "out of memory\n");
    goto end_data_xfer;
  }
  sqlite3_snprintf(200+nTable,zInsert,
                   "INSERT OR IGNORE INTO \"%s\" VALUES(?", zTable);
  i = (int)strlen(zInsert);
  for(j=1; j<n; j++){
    memcpy(zInsert+i, ",?", 2);
    i += 2;
  }
  memcpy(zInsert+i, ");", 3);
  rc = sqlite3_prepare_v2(newDb, zInsert, -1, &pInsert, 0);
  if( rc ){
    utf8_printf(stderr, "Error %d: %s on [%s]\n",
            sqlite3_extended_errcode(newDb), sqlite3_errmsg(newDb),
            zQuery);
    goto end_data_xfer;
  }
  for(k=0; k<2; k++){
    while( (rc = sqlite3_step(pQuery))==SQLITE_ROW ){
      for(i=0; i<n; i++){
        switch( sqlite3_column_type(pQuery, i) ){
          case SQLITE_NULL: {
            sqlite3_bind_null(pInsert, i+1);
            break;
          }
          case SQLITE_INTEGER: {
            sqlite3_bind_int64(pInsert, i+1, sqlite3_column_int64(pQuery,i));
            break;
          }
          case SQLITE_FLOAT: {
            sqlite3_bind_double(pInsert, i+1, sqlite3_column_double(pQuery,i));
            break;
          }
          case SQLITE_TEXT: {
            sqlite3_bind_text(pInsert, i+1,
                             (const char*)sqlite3_column_text(pQuery,i),
                             -1, SQLITE_STATIC);
            break;
          }
          case SQLITE_BLOB: {
            sqlite3_bind_blob(pInsert, i+1, sqlite3_column_blob(pQuery,i),
                                            sqlite3_column_bytes(pQuery,i),
                                            SQLITE_STATIC);
            break;
          }
        }
      } /* End for */
      rc = sqlite3_step(pInsert);
      if( rc!=SQLITE_OK && rc!=SQLITE_ROW && rc!=SQLITE_DONE ){
        utf8_printf(stderr, "Error %d: %s\n", sqlite3_extended_errcode(newDb),
                        sqlite3_errmsg(newDb));
      }
      sqlite3_reset(pInsert);
      cnt++;
      if( (cnt%spinRate)==0 ){
        printf("%c\b", "|/-\\"[(cnt/spinRate)%4]);
        fflush(stdout);
      }
    } /* End while */
    if( rc==SQLITE_DONE ) break;
    sqlite3_finalize(pQuery);
    sqlite3_free(zQuery);
    zQuery = sqlite3_mprintf("SELECT * FROM \"%w\" ORDER BY rowid DESC;",
                             zTable);
    rc = sqlite3_prepare_v2(p->db, zQuery, -1, &pQuery, 0);
    if( rc ){
      utf8_printf(stderr, "Warning: cannot step \"%s\" backwards", zTable);
      break;
    }
  } /* End for(k=0...) */

end_data_xfer:
  sqlite3_finalize(pQuery);
  sqlite3_finalize(pInsert);
  sqlite3_free(zQuery);
  sqlite3_free(zInsert);
}


/*
** Try to transfer all rows of the schema that match zWhere.  For
** each row, invoke xForEach() on the object defined by that row.
** If an error is encountered while moving forward through the
** sqlite_master table, try again moving backwards.
*/
static void tryToCloneSchema(
  ShellState *p,
  sqlite3 *newDb,
  const char *zWhere,
  void (*xForEach)(ShellState*,sqlite3*,const char*)
){
  sqlite3_stmt *pQuery = 0;
  char *zQuery = 0;
  int rc;
  const unsigned char *zName;
  const unsigned char *zSql;
  char *zErrMsg = 0;

  zQuery = sqlite3_mprintf("SELECT name, sql FROM sqlite_master"
                           " WHERE %s", zWhere);
  rc = sqlite3_prepare_v2(p->db, zQuery, -1, &pQuery, 0);
  if( rc ){
    utf8_printf(stderr, "Error: (%d) %s on [%s]\n",
                    sqlite3_extended_errcode(p->db), sqlite3_errmsg(p->db),
                    zQuery);
    goto end_schema_xfer;
  }
  while( (rc = sqlite3_step(pQuery))==SQLITE_ROW ){
    zName = sqlite3_column_text(pQuery, 0);
    zSql = sqlite3_column_text(pQuery, 1);
    printf("%s... ", zName); fflush(stdout);
    sqlite3_exec(newDb, (const char*)zSql, 0, 0, &zErrMsg);
    if( zErrMsg ){
      utf8_printf(stderr, "Error: %s\nSQL: [%s]\n", zErrMsg, zSql);
      sqlite3_free(zErrMsg);
      zErrMsg = 0;
    }
    if( xForEach ){
      xForEach(p, newDb, (const char*)zName);
    }
    printf("done\n");
  }
  if( rc!=SQLITE_DONE ){
    sqlite3_finalize(pQuery);
    sqlite3_free(zQuery);
    zQuery = sqlite3_mprintf("SELECT name, sql FROM sqlite_master"
                             " WHERE %s ORDER BY rowid DESC", zWhere);
    rc = sqlite3_prepare_v2(p->db, zQuery, -1, &pQuery, 0);
    if( rc ){
      utf8_printf(stderr, "Error: (%d) %s on [%s]\n",
                      sqlite3_extended_errcode(p->db), sqlite3_errmsg(p->db),
                      zQuery);
      goto end_schema_xfer;
    }
    while( (rc = sqlite3_step(pQuery))==SQLITE_ROW ){
      zName = sqlite3_column_text(pQuery, 0);
      zSql = sqlite3_column_text(pQuery, 1);
      printf("%s... ", zName); fflush(stdout);
      sqlite3_exec(newDb, (const char*)zSql, 0, 0, &zErrMsg);
      if( zErrMsg ){
        utf8_printf(stderr, "Error: %s\nSQL: [%s]\n", zErrMsg, zSql);
        sqlite3_free(zErrMsg);
        zErrMsg = 0;
      }
      if( xForEach ){
        xForEach(p, newDb, (const char*)zName);
      }
      printf("done\n");
    }
  }
end_schema_xfer:
  sqlite3_finalize(pQuery);
  sqlite3_free(zQuery);
}

/*
** Open a new database file named "zNewDb".  Try to recover as much information
** as possible out of the main database (which might be corrupt) and write it
** into zNewDb.
*/
static void tryToClone(ShellState *p, const char *zNewDb){
  int rc;
  sqlite3 *newDb = 0;
  if( access(zNewDb,0)==0 ){
    utf8_printf(stderr, "File \"%s\" already exists.\n", zNewDb);
    return;
  }
  rc = sqlite3_open(zNewDb, &newDb);
  if( rc ){
    utf8_printf(stderr, "Cannot create output database: %s\n",
            sqlite3_errmsg(newDb));
  }else{
    sqlite3_exec(p->db, "PRAGMA writable_schema=ON;", 0, 0, 0);
    sqlite3_exec(newDb, "BEGIN EXCLUSIVE;", 0, 0, 0);
    tryToCloneSchema(p, newDb, "type='table'", tryToCloneData);
    tryToCloneSchema(p, newDb, "type!='table'", 0);
    sqlite3_exec(newDb, "COMMIT;", 0, 0, 0);
    sqlite3_exec(p->db, "PRAGMA writable_schema=OFF;", 0, 0, 0);
  }
  sqlite3_close(newDb);
}

/*
** Change the output file back to stdout
*/
static void output_reset(ShellState *p){
  if( p->outfile[0]=='|' ){
#ifndef SQLITE_OMIT_POPEN
    pclose(p->out);
#endif
  }else{
    output_file_close(p->out);
  }
  p->outfile[0] = 0;
  p->out = stdout;
}

/*
** Run an SQL command and return the single integer result.
*/
static int db_int(ShellState *p, const char *zSql){
  sqlite3_stmt *pStmt;
  int res = 0;
  sqlite3_prepare_v2(p->db, zSql, -1, &pStmt, 0);
  if( pStmt && sqlite3_step(pStmt)==SQLITE_ROW ){
    res = sqlite3_column_int(pStmt,0);
  }
  sqlite3_finalize(pStmt);
  return res;
}

/*
** Convert a 2-byte or 4-byte big-endian integer into a native integer
*/
static unsigned int get2byteInt(unsigned char *a){
  return (a[0]<<8) + a[1];
}
static unsigned int get4byteInt(unsigned char *a){
  return (a[0]<<24) + (a[1]<<16) + (a[2]<<8) + a[3];
}

/*
** Implementation of the ".info" command.
**
** Return 1 on error, 2 to exit, and 0 otherwise.
*/
static int shell_dbinfo_command(ShellState *p, int nArg, char **azArg){
  static const struct { const char *zName; int ofst; } aField[] = {
     { "file change counter:",  24  },
     { "database page count:",  28  },
     { "freelist page count:",  36  },
     { "schema cookie:",        40  },
     { "schema format:",        44  },
     { "default cache size:",   48  },
     { "autovacuum top root:",  52  },
     { "incremental vacuum:",   64  },
     { "text encoding:",        56  },
     { "user version:",         60  },
     { "application id:",       68  },
     { "software version:",     96  },
  };
  static const struct { const char *zName; const char *zSql; } aQuery[] = {
     { "number of tables:",
       "SELECT count(*) FROM %s WHERE type='table'" },
     { "number of indexes:",
       "SELECT count(*) FROM %s WHERE type='index'" },
     { "number of triggers:",
       "SELECT count(*) FROM %s WHERE type='trigger'" },
     { "number of views:",
       "SELECT count(*) FROM %s WHERE type='view'" },
     { "schema size:",
       "SELECT total(length(sql)) FROM %s" },
  };
  sqlite3_file *pFile = 0;
  int i;
  char *zSchemaTab;
  char *zDb = nArg>=2 ? azArg[1] : "main";
  unsigned char aHdr[100];
  open_db(p, 0);
  if( p->db==0 ) return 1;
  sqlite3_file_control(p->db, zDb, SQLITE_FCNTL_FILE_POINTER, &pFile);
  if( pFile==0 || pFile->pMethods==0 || pFile->pMethods->xRead==0 ){
    return 1;
  }
  i = pFile->pMethods->xRead(pFile, aHdr, 100, 0);
  if( i!=SQLITE_OK ){
    raw_printf(stderr, "unable to read database header\n");
    return 1;
  }
  i = get2byteInt(aHdr+16);
  if( i==1 ) i = 65536;
  utf8_printf(p->out, "%-20s %d\n", "database page size:", i);
  utf8_printf(p->out, "%-20s %d\n", "write format:", aHdr[18]);
  utf8_printf(p->out, "%-20s %d\n", "read format:", aHdr[19]);
  utf8_printf(p->out, "%-20s %d\n", "reserved bytes:", aHdr[20]);
  for(i=0; i<ArraySize(aField); i++){
    int ofst = aField[i].ofst;
    unsigned int val = get4byteInt(aHdr + ofst);
    utf8_printf(p->out, "%-20s %u", aField[i].zName, val);
    switch( ofst ){
      case 56: {
        if( val==1 ) raw_printf(p->out, " (utf8)");
        if( val==2 ) raw_printf(p->out, " (utf16le)");
        if( val==3 ) raw_printf(p->out, " (utf16be)");
      }
    }
    raw_printf(p->out, "\n");
  }
  if( zDb==0 ){
    zSchemaTab = sqlite3_mprintf("main.sqlite_master");
  }else if( strcmp(zDb,"temp")==0 ){
    zSchemaTab = sqlite3_mprintf("%s", "sqlite_temp_master");
  }else{
    zSchemaTab = sqlite3_mprintf("\"%w\".sqlite_master", zDb);
  }
  for(i=0; i<ArraySize(aQuery); i++){
    char *zSql = sqlite3_mprintf(aQuery[i].zSql, zSchemaTab);
    int val = db_int(p, zSql);
    sqlite3_free(zSql);
    utf8_printf(p->out, "%-20s %d\n", aQuery[i].zName, val);
  }
  sqlite3_free(zSchemaTab);
  return 0;
}

/*
** Print the current sqlite3_errmsg() value to stderr and return 1.
*/
static int shellDatabaseError(sqlite3 *db){
  const char *zErr = sqlite3_errmsg(db);
  utf8_printf(stderr, "Error: %s\n", zErr);
  return 1;
}

/*
** Print an out-of-memory message to stderr and return 1.
*/
static int shellNomemError(void){
  raw_printf(stderr, "Error: out of memory\n");
  return 1;
}

/*
** Compare the pattern in zGlob[] against the text in z[].  Return TRUE
** if they match and FALSE (0) if they do not match.
**
** Globbing rules:
**
**      '*'       Matches any sequence of zero or more characters.
**
**      '?'       Matches exactly one character.
**
**     [...]      Matches one character from the enclosed list of
**                characters.
**
**     [^...]     Matches one character not in the enclosed list.
**
**      '#'       Matches any sequence of one or more digits with an
**                optional + or - sign in front
**
**      ' '       Any span of whitespace matches any other span of
**                whitespace.
**
** Extra whitespace at the end of z[] is ignored.
*/
static int testcase_glob(const char *zGlob, const char *z){
  int c, c2;
  int invert;
  int seen;

  while( (c = (*(zGlob++)))!=0 ){
    if( IsSpace(c) ){
      if( !IsSpace(*z) ) return 0;
      while( IsSpace(*zGlob) ) zGlob++;
      while( IsSpace(*z) ) z++;
    }else if( c=='*' ){
      while( (c=(*(zGlob++))) == '*' || c=='?' ){
        if( c=='?' && (*(z++))==0 ) return 0;
      }
      if( c==0 ){
        return 1;
      }else if( c=='[' ){
        while( *z && testcase_glob(zGlob-1,z)==0 ){
          z++;
        }
        return (*z)!=0;
      }
      while( (c2 = (*(z++)))!=0 ){
        while( c2!=c ){
          c2 = *(z++);
          if( c2==0 ) return 0;
        }
        if( testcase_glob(zGlob,z) ) return 1;
      }
      return 0;
    }else if( c=='?' ){
      if( (*(z++))==0 ) return 0;
    }else if( c=='[' ){
      int prior_c = 0;
      seen = 0;
      invert = 0;
      c = *(z++);
      if( c==0 ) return 0;
      c2 = *(zGlob++);
      if( c2=='^' ){
        invert = 1;
        c2 = *(zGlob++);
      }
      if( c2==']' ){
        if( c==']' ) seen = 1;
        c2 = *(zGlob++);
      }
      while( c2 && c2!=']' ){
        if( c2=='-' && zGlob[0]!=']' && zGlob[0]!=0 && prior_c>0 ){
          c2 = *(zGlob++);
          if( c>=prior_c && c<=c2 ) seen = 1;
          prior_c = 0;
        }else{
          if( c==c2 ){
            seen = 1;
          }
          prior_c = c2;
        }
        c2 = *(zGlob++);
      }
      if( c2==0 || (seen ^ invert)==0 ) return 0;
    }else if( c=='#' ){
      if( (z[0]=='-' || z[0]=='+') && IsDigit(z[1]) ) z++;
      if( !IsDigit(z[0]) ) return 0;
      z++;
      while( IsDigit(z[0]) ){ z++; }
    }else{
      if( c!=(*(z++)) ) return 0;
    }
  }
  while( IsSpace(*z) ){ z++; }
  return *z==0;
}


/*
** Compare the string as a command-line option with either one or two
** initial "-" characters.
*/
static int optionMatch(const char *zStr, const char *zOpt){
  if( zStr[0]!='-' ) return 0;
  zStr++;
  if( zStr[0]=='-' ) zStr++;
  return strcmp(zStr, zOpt)==0;
}

/*
** Delete a file.
*/
int shellDeleteFile(const char *zFilename){
  int rc;
#ifdef _WIN32
  wchar_t *z = sqlite3_win32_utf8_to_unicode(zFilename);
  rc = _wunlink(z);
  sqlite3_free(z);
#else
  rc = unlink(zFilename);
#endif
  return rc;
}


/*
** The implementation of SQL scalar function fkey_collate_clause(), used
** by the ".lint fkey-indexes" command. This scalar function is always
** called with four arguments - the parent table name, the parent column name,
** the child table name and the child column name.
**
**   fkey_collate_clause('parent-tab', 'parent-col', 'child-tab', 'child-col')
**
** If either of the named tables or columns do not exist, this function
** returns an empty string. An empty string is also returned if both tables 
** and columns exist but have the same default collation sequence. Or,
** if both exist but the default collation sequences are different, this
** function returns the string " COLLATE <parent-collation>", where
** <parent-collation> is the default collation sequence of the parent column.
*/
static void shellFkeyCollateClause(
  sqlite3_context *pCtx, 
  int nVal, 
  sqlite3_value **apVal
){
  sqlite3 *db = sqlite3_context_db_handle(pCtx);
  const char *zParent;
  const char *zParentCol;
  const char *zParentSeq;
  const char *zChild;
  const char *zChildCol;
  const char *zChildSeq = 0;  /* Initialize to avoid false-positive warning */
  int rc;
  
  assert( nVal==4 );
  zParent = (const char*)sqlite3_value_text(apVal[0]);
  zParentCol = (const char*)sqlite3_value_text(apVal[1]);
  zChild = (const char*)sqlite3_value_text(apVal[2]);
  zChildCol = (const char*)sqlite3_value_text(apVal[3]);

  sqlite3_result_text(pCtx, "", -1, SQLITE_STATIC);
  rc = sqlite3_table_column_metadata(
      db, "main", zParent, zParentCol, 0, &zParentSeq, 0, 0, 0
  );
  if( rc==SQLITE_OK ){
    rc = sqlite3_table_column_metadata(
        db, "main", zChild, zChildCol, 0, &zChildSeq, 0, 0, 0
    );
  }

  if( rc==SQLITE_OK && sqlite3_stricmp(zParentSeq, zChildSeq) ){
    char *z = sqlite3_mprintf(" COLLATE %s", zParentSeq);
    sqlite3_result_text(pCtx, z, -1, SQLITE_TRANSIENT);
    sqlite3_free(z);
  }
}


/*
** The implementation of dot-command ".lint fkey-indexes".
*/
static int lintFkeyIndexes(
  ShellState *pState,             /* Current shell tool state */
  char **azArg,                   /* Array of arguments passed to dot command */
  int nArg                        /* Number of entries in azArg[] */
){
  sqlite3 *db = pState->db;       /* Database handle to query "main" db of */
  FILE *out = pState->out;        /* Stream to write non-error output to */
  int bVerbose = 0;               /* If -verbose is present */
  int bGroupByParent = 0;         /* If -groupbyparent is present */
  int i;                          /* To iterate through azArg[] */
  const char *zIndent = "";       /* How much to indent CREATE INDEX by */
  int rc;                         /* Return code */
  sqlite3_stmt *pSql = 0;         /* Compiled version of SQL statement below */

  /*
  ** This SELECT statement returns one row for each foreign key constraint
  ** in the schema of the main database. The column values are:
  **
  ** 0. The text of an SQL statement similar to:
  **
  **      "EXPLAIN QUERY PLAN SELECT rowid FROM child_table WHERE child_key=?"
  **
  **    This is the same SELECT that the foreign keys implementation needs
  **    to run internally on child tables. If there is an index that can
  **    be used to optimize this query, then it can also be used by the FK
  **    implementation to optimize DELETE or UPDATE statements on the parent
  **    table.
  **
  ** 1. A GLOB pattern suitable for sqlite3_strglob(). If the plan output by
  **    the EXPLAIN QUERY PLAN command matches this pattern, then the schema
  **    contains an index that can be used to optimize the query.
  **
  ** 2. Human readable text that describes the child table and columns. e.g.
  **
  **       "child_table(child_key1, child_key2)"
  **
  ** 3. Human readable text that describes the parent table and columns. e.g.
  **
  **       "parent_table(parent_key1, parent_key2)"
  **
  ** 4. A full CREATE INDEX statement for an index that could be used to
  **    optimize DELETE or UPDATE statements on the parent table. e.g.
  **
  **       "CREATE INDEX child_table_child_key ON child_table(child_key)"
  **
  ** 5. The name of the parent table.
  **
  ** These six values are used by the C logic below to generate the report.
  */
  const char *zSql =
  "SELECT "
    "     'EXPLAIN QUERY PLAN SELECT rowid FROM ' || quote(s.name) || ' WHERE '"
    "  || group_concat(quote(s.name) || '.' || quote(f.[from]) || '=?' "
    "  || fkey_collate_clause(f.[table], f.[to], s.name, f.[from]),' AND ')"
    ", "
    "     'SEARCH TABLE ' || s.name || ' USING COVERING INDEX*('"
    "  || group_concat('*=?', ' AND ') || ')'"
    ", "
    "     s.name  || '(' || group_concat(f.[from],  ', ') || ')'"
    ", "
    "     f.[table] || '(' || group_concat(COALESCE(f.[to], "
    "       (SELECT name FROM pragma_table_info(f.[table]) WHERE pk=seq+1)"
    "     )) || ')'"
    ", "
    "     'CREATE INDEX ' || quote(s.name ||'_'|| group_concat(f.[from], '_'))"
    "  || ' ON ' || quote(s.name) || '('"
    "  || group_concat(quote(f.[from]) ||"
    "        fkey_collate_clause(f.[table], f.[to], s.name, f.[from]), ', ')"
    "  || ');'"
    ", "
    "     f.[table] "

    "FROM sqlite_master AS s, pragma_foreign_key_list(s.name) AS f "
    "GROUP BY s.name, f.id "
    "ORDER BY (CASE WHEN ? THEN f.[table] ELSE s.name END)"
  ;

  for(i=2; i<nArg; i++){
    int n = (int)strlen(azArg[i]);
    if( n>1 && sqlite3_strnicmp("-verbose", azArg[i], n)==0 ){
      bVerbose = 1;
    }
    else if( n>1 && sqlite3_strnicmp("-groupbyparent", azArg[i], n)==0 ){
      bGroupByParent = 1;
      zIndent = "    ";
    }
    else{
      raw_printf(stderr, "Usage: %s %s ?-verbose? ?-groupbyparent?\n",
          azArg[0], azArg[1]
      );
      return SQLITE_ERROR;
    }
  }
  
  /* Register the fkey_collate_clause() SQL function */
  rc = sqlite3_create_function(db, "fkey_collate_clause", 4, SQLITE_UTF8,
      0, shellFkeyCollateClause, 0, 0
  );


  if( rc==SQLITE_OK ){
    rc = sqlite3_prepare_v2(db, zSql, -1, &pSql, 0);
  }
  if( rc==SQLITE_OK ){
    sqlite3_bind_int(pSql, 1, bGroupByParent);
  }

  if( rc==SQLITE_OK ){
    int rc2;
    char *zPrev = 0;
    while( SQLITE_ROW==sqlite3_step(pSql) ){
      int res = -1;
      sqlite3_stmt *pExplain = 0;
      const char *zEQP = (const char*)sqlite3_column_text(pSql, 0);
      const char *zGlob = (const char*)sqlite3_column_text(pSql, 1);
      const char *zFrom = (const char*)sqlite3_column_text(pSql, 2);
      const char *zTarget = (const char*)sqlite3_column_text(pSql, 3);
      const char *zCI = (const char*)sqlite3_column_text(pSql, 4);
      const char *zParent = (const char*)sqlite3_column_text(pSql, 5);

      rc = sqlite3_prepare_v2(db, zEQP, -1, &pExplain, 0);
      if( rc!=SQLITE_OK ) break;
      if( SQLITE_ROW==sqlite3_step(pExplain) ){
        const char *zPlan = (const char*)sqlite3_column_text(pExplain, 3);
        res = (0==sqlite3_strglob(zGlob, zPlan));
      }
      rc = sqlite3_finalize(pExplain);
      if( rc!=SQLITE_OK ) break;

      if( res<0 ){
        raw_printf(stderr, "Error: internal error");
        break;
      }else{
        if( bGroupByParent 
        && (bVerbose || res==0)
        && (zPrev==0 || sqlite3_stricmp(zParent, zPrev)) 
        ){
          raw_printf(out, "-- Parent table %s\n", zParent);
          sqlite3_free(zPrev);
          zPrev = sqlite3_mprintf("%s", zParent);
        }

        if( res==0 ){
          raw_printf(out, "%s%s --> %s\n", zIndent, zCI, zTarget);
        }else if( bVerbose ){
          raw_printf(out, "%s/* no extra indexes required for %s -> %s */\n", 
              zIndent, zFrom, zTarget
          );
        }
      }
    }
    sqlite3_free(zPrev);

    if( rc!=SQLITE_OK ){
      raw_printf(stderr, "%s\n", sqlite3_errmsg(db));
    }

    rc2 = sqlite3_finalize(pSql);
    if( rc==SQLITE_OK && rc2!=SQLITE_OK ){
      rc = rc2;
      raw_printf(stderr, "%s\n", sqlite3_errmsg(db));
    }
  }else{
    raw_printf(stderr, "%s\n", sqlite3_errmsg(db));
  }

  return rc;
}

/*
** Implementation of ".lint" dot command.
*/
static int lintDotCommand(
  ShellState *pState,             /* Current shell tool state */
  char **azArg,                   /* Array of arguments passed to dot command */
  int nArg                        /* Number of entries in azArg[] */
){
  int n;
  n = (nArg>=2 ? (int)strlen(azArg[1]) : 0);
  if( n<1 || sqlite3_strnicmp(azArg[1], "fkey-indexes", n) ) goto usage;
  return lintFkeyIndexes(pState, azArg, nArg);

 usage:
  raw_printf(stderr, "Usage %s sub-command ?switches...?\n", azArg[0]);
  raw_printf(stderr, "Where sub-commands are:\n");
  raw_printf(stderr, "    fkey-indexes\n");
  return SQLITE_ERROR;
}


/*
** If an input line begins with "." then invoke this routine to
** process that line.
**
** Return 1 on error, 2 to exit, and 0 otherwise.
*/
static int do_meta_command(char *zLine, ShellState *p){
  int h = 1;
  int nArg = 0;
  int n, c;
  int rc = 0;
  char *azArg[50];

  /* Parse the input line into tokens.
  */
  while( zLine[h] && nArg<ArraySize(azArg) ){
    while( IsSpace(zLine[h]) ){ h++; }
    if( zLine[h]==0 ) break;
    if( zLine[h]=='\'' || zLine[h]=='"' ){
      int delim = zLine[h++];
      azArg[nArg++] = &zLine[h];
      while( zLine[h] && zLine[h]!=delim ){
        if( zLine[h]=='\\' && delim=='"' && zLine[h+1]!=0 ) h++;
        h++;
      }
      if( zLine[h]==delim ){
        zLine[h++] = 0;
      }
      if( delim=='"' ) resolve_backslashes(azArg[nArg-1]);
    }else{
      azArg[nArg++] = &zLine[h];
      while( zLine[h] && !IsSpace(zLine[h]) ){ h++; }
      if( zLine[h] ) zLine[h++] = 0;
      resolve_backslashes(azArg[nArg-1]);
    }
  }

  /* Process the input line.
  */
  if( nArg==0 ) return 0; /* no tokens, no error */
  n = strlen30(azArg[0]);
  c = azArg[0][0];

#ifndef SQLITE_OMIT_AUTHORIZATION
  if( c=='a' && strncmp(azArg[0], "auth", n)==0 ){
    if( nArg!=2 ){
      raw_printf(stderr, "Usage: .auth ON|OFF\n");
      rc = 1;
      goto meta_command_exit;
    }
    open_db(p, 0);
    if( booleanValue(azArg[1]) ){
      sqlite3_set_authorizer(p->db, shellAuth, p);
    }else{
      sqlite3_set_authorizer(p->db, 0, 0);
    }
  }else
#endif

  if( (c=='b' && n>=3 && strncmp(azArg[0], "backup", n)==0)
   || (c=='s' && n>=3 && strncmp(azArg[0], "save", n)==0)
  ){
    const char *zDestFile = 0;
    const char *zDb = 0;
    sqlite3 *pDest;
    sqlite3_backup *pBackup;
    int j;
    for(j=1; j<nArg; j++){
      const char *z = azArg[j];
      if( z[0]=='-' ){
        while( z[0]=='-' ) z++;
        /* No options to process at this time */
        {
          utf8_printf(stderr, "unknown option: %s\n", azArg[j]);
          return 1;
        }
      }else if( zDestFile==0 ){
        zDestFile = azArg[j];
      }else if( zDb==0 ){
        zDb = zDestFile;
        zDestFile = azArg[j];
      }else{
        raw_printf(stderr, "too many arguments to .backup\n");
        return 1;
      }
    }
    if( zDestFile==0 ){
      raw_printf(stderr, "missing FILENAME argument on .backup\n");
      return 1;
    }
    if( zDb==0 ) zDb = "main";
    rc = sqlite3_open(zDestFile, &pDest);
    if( rc!=SQLITE_OK ){
      utf8_printf(stderr, "Error: cannot open \"%s\"\n", zDestFile);
      sqlite3_close(pDest);
      return 1;
    }
    open_db(p, 0);
    pBackup = sqlite3_backup_init(pDest, "main", p->db, zDb);
    if( pBackup==0 ){
      utf8_printf(stderr, "Error: %s\n", sqlite3_errmsg(pDest));
      sqlite3_close(pDest);
      return 1;
    }
    while(  (rc = sqlite3_backup_step(pBackup,100))==SQLITE_OK ){}
    sqlite3_backup_finish(pBackup);
    if( rc==SQLITE_DONE ){
      rc = 0;
    }else{
      utf8_printf(stderr, "Error: %s\n", sqlite3_errmsg(pDest));
      rc = 1;
    }
    sqlite3_close(pDest);
  }else

  if( c=='b' && n>=3 && strncmp(azArg[0], "bail", n)==0 ){
    if( nArg==2 ){
      bail_on_error = booleanValue(azArg[1]);
    }else{
      raw_printf(stderr, "Usage: .bail on|off\n");
      rc = 1;
    }
  }else

  if( c=='b' && n>=3 && strncmp(azArg[0], "binary", n)==0 ){
    if( nArg==2 ){
      if( booleanValue(azArg[1]) ){
        setBinaryMode(p->out, 1);
      }else{
        setTextMode(p->out, 1);
      }
    }else{
      raw_printf(stderr, "Usage: .binary on|off\n");
      rc = 1;
    }
  }else

  /* The undocumented ".breakpoint" command causes a call to the no-op
  ** routine named test_breakpoint().
  */
  if( c=='b' && n>=3 && strncmp(azArg[0], "breakpoint", n)==0 ){
    test_breakpoint();
  }else

  if( c=='c' && n>=3 && strncmp(azArg[0], "changes", n)==0 ){
    if( nArg==2 ){
      setOrClearFlag(p, SHFLG_CountChanges, azArg[1]);
    }else{
      raw_printf(stderr, "Usage: .changes on|off\n");
      rc = 1;
    }
  }else

  /* Cancel output redirection, if it is currently set (by .testcase)
  ** Then read the content of the testcase-out.txt file and compare against
  ** azArg[1].  If there are differences, report an error and exit.
  */
  if( c=='c' && n>=3 && strncmp(azArg[0], "check", n)==0 ){
    char *zRes = 0;
    output_reset(p);
    if( nArg!=2 ){
      raw_printf(stderr, "Usage: .check GLOB-PATTERN\n");
      rc = 2;
    }else if( (zRes = readFile("testcase-out.txt", 0))==0 ){
      raw_printf(stderr, "Error: cannot read 'testcase-out.txt'\n");
      rc = 2;
    }else if( testcase_glob(azArg[1],zRes)==0 ){
      utf8_printf(stderr,
                 "testcase-%s FAILED\n Expected: [%s]\n      Got: [%s]\n",
                 p->zTestcase, azArg[1], zRes);
      rc = 2;
    }else{
      utf8_printf(stdout, "testcase-%s ok\n", p->zTestcase);
      p->nCheck++;
    }
    sqlite3_free(zRes);
  }else

  if( c=='c' && strncmp(azArg[0], "clone", n)==0 ){
    if( nArg==2 ){
      tryToClone(p, azArg[1]);
    }else{
      raw_printf(stderr, "Usage: .clone FILENAME\n");
      rc = 1;
    }
  }else

  if( c=='d' && n>1 && strncmp(azArg[0], "databases", n)==0 ){
    ShellState data;
    char *zErrMsg = 0;
    open_db(p, 0);
    memcpy(&data, p, sizeof(data));
    data.showHeader = 0;
    data.cMode = data.mode = MODE_List;
    sqlite3_snprintf(sizeof(data.colSeparator),data.colSeparator,": ");
    data.cnt = 0;
    sqlite3_exec(p->db, "SELECT name, file FROM pragma_database_list",
                 callback, &data, &zErrMsg);
    if( zErrMsg ){
      utf8_printf(stderr,"Error: %s\n", zErrMsg);
      sqlite3_free(zErrMsg);
      rc = 1;
    }
  }else

  if( c=='d' && strncmp(azArg[0], "dbinfo", n)==0 ){
    rc = shell_dbinfo_command(p, nArg, azArg);
  }else

  if( c=='d' && strncmp(azArg[0], "dump", n)==0 ){
    const char *zLike = 0;
    int i;
    ShellClearFlag(p, SHFLG_PreserveRowid);
    for(i=1; i<nArg; i++){
      if( azArg[i][0]=='-' ){
        const char *z = azArg[i]+1;
        if( z[0]=='-' ) z++;
        if( strcmp(z,"preserve-rowids")==0 ){
#ifdef SQLITE_OMIT_VIRTUALTABLE
          raw_printf(stderr, "The --preserve-rowids option is not compatible"
                             " with SQLITE_OMIT_VIRTUALTABLE\n");
          rc = 1;
          goto meta_command_exit;
#else
          ShellSetFlag(p, SHFLG_PreserveRowid);
#endif
        }else
        {
          raw_printf(stderr, "Unknown option \"%s\" on \".dump\"\n", azArg[i]);
          rc = 1;
          goto meta_command_exit;
        }
      }else if( zLike ){
        raw_printf(stderr, "Usage: .dump ?--preserve-rowids? ?LIKE-PATTERN?\n");
        rc = 1;
        goto meta_command_exit;
      }else{
        zLike = azArg[i];
      }
    }
    open_db(p, 0);
    /* When playing back a "dump", the content might appear in an order
    ** which causes immediate foreign key constraints to be violated.
    ** So disable foreign-key constraint enforcement to prevent problems. */
    raw_printf(p->out, "PRAGMA foreign_keys=OFF;\n");
    raw_printf(p->out, "BEGIN TRANSACTION;\n");
    p->writableSchema = 0;
    /* Set writable_schema=ON since doing so forces SQLite to initialize
    ** as much of the schema as it can even if the sqlite_master table is
    ** corrupt. */
    sqlite3_exec(p->db, "SAVEPOINT dump; PRAGMA writable_schema=ON", 0, 0, 0);
    p->nErr = 0;
    if( zLike==0 ){
      run_schema_dump_query(p,
        "SELECT name, type, sql FROM sqlite_master "
        "WHERE sql NOT NULL AND type=='table' AND name!='sqlite_sequence'"
      );
      run_schema_dump_query(p,
        "SELECT name, type, sql FROM sqlite_master "
        "WHERE name=='sqlite_sequence'"
      );
      run_table_dump_query(p,
        "SELECT sql FROM sqlite_master "
        "WHERE sql NOT NULL AND type IN ('index','trigger','view')", 0
      );
    }else{
      char *zSql;
      zSql = sqlite3_mprintf(
        "SELECT name, type, sql FROM sqlite_master "
        "WHERE tbl_name LIKE %Q AND type=='table'"
        "  AND sql NOT NULL", zLike);
      run_schema_dump_query(p,zSql);
      sqlite3_free(zSql);
      zSql = sqlite3_mprintf(
        "SELECT sql FROM sqlite_master "
        "WHERE sql NOT NULL"
        "  AND type IN ('index','trigger','view')"
        "  AND tbl_name LIKE %Q", zLike);
      run_table_dump_query(p, zSql, 0);
      sqlite3_free(zSql);
    }
    if( p->writableSchema ){
      raw_printf(p->out, "PRAGMA writable_schema=OFF;\n");
      p->writableSchema = 0;
    }
    sqlite3_exec(p->db, "PRAGMA writable_schema=OFF;", 0, 0, 0);
    sqlite3_exec(p->db, "RELEASE dump;", 0, 0, 0);
    raw_printf(p->out, p->nErr ? "ROLLBACK; -- due to errors\n" : "COMMIT;\n");
  }else

  if( c=='e' && strncmp(azArg[0], "echo", n)==0 ){
    if( nArg==2 ){
      setOrClearFlag(p, SHFLG_Echo, azArg[1]);
    }else{
      raw_printf(stderr, "Usage: .echo on|off\n");
      rc = 1;
    }
  }else

  if( c=='e' && strncmp(azArg[0], "eqp", n)==0 ){
    if( nArg==2 ){
      if( strcmp(azArg[1],"full")==0 ){
        p->autoEQP = 2;
      }else{
        p->autoEQP = booleanValue(azArg[1]);
      }
    }else{
      raw_printf(stderr, "Usage: .eqp on|off|full\n");
      rc = 1;
    }
  }else

  if( c=='e' && strncmp(azArg[0], "exit", n)==0 ){
    if( nArg>1 && (rc = (int)integerValue(azArg[1]))!=0 ) exit(rc);
    rc = 2;
  }else

  if( c=='e' && strncmp(azArg[0], "explain", n)==0 ){
    int val = 1;
    if( nArg>=2 ){
      if( strcmp(azArg[1],"auto")==0 ){
        val = 99;
      }else{
        val =  booleanValue(azArg[1]);
      }
    }
    if( val==1 && p->mode!=MODE_Explain ){
      p->normalMode = p->mode;
      p->mode = MODE_Explain;
      p->autoExplain = 0;
    }else if( val==0 ){
      if( p->mode==MODE_Explain ) p->mode = p->normalMode;
      p->autoExplain = 0;
    }else if( val==99 ){
      if( p->mode==MODE_Explain ) p->mode = p->normalMode;
      p->autoExplain = 1;
    }
  }else

  if( c=='f' && strncmp(azArg[0], "fullschema", n)==0 ){
    ShellState data;
    char *zErrMsg = 0;
    int doStats = 0;
    memcpy(&data, p, sizeof(data));
    data.showHeader = 0;
    data.cMode = data.mode = MODE_Semi;
    if( nArg==2 && optionMatch(azArg[1], "indent") ){
      data.cMode = data.mode = MODE_Pretty;
      nArg = 1;
    }
    if( nArg!=1 ){
      raw_printf(stderr, "Usage: .fullschema ?--indent?\n");
      rc = 1;
      goto meta_command_exit;
    }
    open_db(p, 0);
    rc = sqlite3_exec(p->db,
       "SELECT sql FROM"
       "  (SELECT sql sql, type type, tbl_name tbl_name, name name, rowid x"
       "     FROM sqlite_master UNION ALL"
       "   SELECT sql, type, tbl_name, name, rowid FROM sqlite_temp_master) "
       "WHERE type!='meta' AND sql NOTNULL AND name NOT LIKE 'sqlite_%' "
       "ORDER BY rowid",
       callback, &data, &zErrMsg
    );
    if( rc==SQLITE_OK ){
      sqlite3_stmt *pStmt;
      rc = sqlite3_prepare_v2(p->db,
               "SELECT rowid FROM sqlite_master"
               " WHERE name GLOB 'sqlite_stat[134]'",
               -1, &pStmt, 0);
      doStats = sqlite3_step(pStmt)==SQLITE_ROW;
      sqlite3_finalize(pStmt);
    }
    if( doStats==0 ){
      raw_printf(p->out, "/* No STAT tables available */\n");
    }else{
      raw_printf(p->out, "ANALYZE sqlite_master;\n");
      sqlite3_exec(p->db, "SELECT 'ANALYZE sqlite_master'",
                   callback, &data, &zErrMsg);
      data.cMode = data.mode = MODE_Insert;
      data.zDestTable = "sqlite_stat1";
      shell_exec(p->db, "SELECT * FROM sqlite_stat1",
                 shell_callback, &data,&zErrMsg);
      data.zDestTable = "sqlite_stat3";
      shell_exec(p->db, "SELECT * FROM sqlite_stat3",
                 shell_callback, &data,&zErrMsg);
      data.zDestTable = "sqlite_stat4";
      shell_exec(p->db, "SELECT * FROM sqlite_stat4",
                 shell_callback, &data, &zErrMsg);
      raw_printf(p->out, "ANALYZE sqlite_master;\n");
    }
  }else

  if( c=='h' && strncmp(azArg[0], "headers", n)==0 ){
    if( nArg==2 ){
      p->showHeader = booleanValue(azArg[1]);
    }else{
      raw_printf(stderr, "Usage: .headers on|off\n");
      rc = 1;
    }
  }else

  if( c=='h' && strncmp(azArg[0], "help", n)==0 ){
    utf8_printf(p->out, "%s", zHelp);
  }else

  if( c=='i' && strncmp(azArg[0], "import", n)==0 ){
    char *zTable;               /* Insert data into this table */
    char *zFile;                /* Name of file to extra content from */
    sqlite3_stmt *pStmt = NULL; /* A statement */
    int nCol;                   /* Number of columns in the table */
    int nByte;                  /* Number of bytes in an SQL string */
    int i, j;                   /* Loop counters */
    int needCommit;             /* True to COMMIT or ROLLBACK at end */
    int nSep;                   /* Number of bytes in p->colSeparator[] */
    char *zSql;                 /* An SQL statement */
    ImportCtx sCtx;             /* Reader context */
    char *(SQLITE_CDECL *xRead)(ImportCtx*); /* Func to read one value */
    int (SQLITE_CDECL *xCloser)(FILE*);      /* Func to close file */

    if( nArg!=3 ){
      raw_printf(stderr, "Usage: .import FILE TABLE\n");
      goto meta_command_exit;
    }
    zFile = azArg[1];
    zTable = azArg[2];
    seenInterrupt = 0;
    memset(&sCtx, 0, sizeof(sCtx));
    open_db(p, 0);
    nSep = strlen30(p->colSeparator);
    if( nSep==0 ){
      raw_printf(stderr,
                 "Error: non-null column separator required for import\n");
      return 1;
    }
    if( nSep>1 ){
      raw_printf(stderr, "Error: multi-character column separators not allowed"
                      " for import\n");
      return 1;
    }
    nSep = strlen30(p->rowSeparator);
    if( nSep==0 ){
      raw_printf(stderr, "Error: non-null row separator required for import\n");
      return 1;
    }
    if( nSep==2 && p->mode==MODE_Csv && strcmp(p->rowSeparator, SEP_CrLf)==0 ){
      /* When importing CSV (only), if the row separator is set to the
      ** default output row separator, change it to the default input
      ** row separator.  This avoids having to maintain different input
      ** and output row separators. */
      sqlite3_snprintf(sizeof(p->rowSeparator), p->rowSeparator, SEP_Row);
      nSep = strlen30(p->rowSeparator);
    }
    if( nSep>1 ){
      raw_printf(stderr, "Error: multi-character row separators not allowed"
                      " for import\n");
      return 1;
    }
    sCtx.zFile = zFile;
    sCtx.nLine = 1;
    if( sCtx.zFile[0]=='|' ){
#ifdef SQLITE_OMIT_POPEN
      raw_printf(stderr, "Error: pipes are not supported in this OS\n");
      return 1;
#else
      sCtx.in = popen(sCtx.zFile+1, "r");
      sCtx.zFile = "<pipe>";
      xCloser = pclose;
#endif
    }else{
      sCtx.in = fopen(sCtx.zFile, "rb");
      xCloser = fclose;
    }
    if( p->mode==MODE_Ascii ){
      xRead = ascii_read_one_field;
    }else{
      xRead = csv_read_one_field;
    }
    if( sCtx.in==0 ){
      utf8_printf(stderr, "Error: cannot open \"%s\"\n", zFile);
      return 1;
    }
    sCtx.cColSep = p->colSeparator[0];
    sCtx.cRowSep = p->rowSeparator[0];
    zSql = sqlite3_mprintf("SELECT * FROM %s", zTable);
    if( zSql==0 ){
      raw_printf(stderr, "Error: out of memory\n");
      xCloser(sCtx.in);
      return 1;
    }
    nByte = strlen30(zSql);
    rc = sqlite3_prepare_v2(p->db, zSql, -1, &pStmt, 0);
    import_append_char(&sCtx, 0);    /* To ensure sCtx.z is allocated */
    if( rc && sqlite3_strglob("no such table: *", sqlite3_errmsg(p->db))==0 ){
      char *zCreate = sqlite3_mprintf("CREATE TABLE %s", zTable);
      char cSep = '(';
      while( xRead(&sCtx) ){
        zCreate = sqlite3_mprintf("%z%c\n  \"%w\" TEXT", zCreate, cSep, sCtx.z);
        cSep = ',';
        if( sCtx.cTerm!=sCtx.cColSep ) break;
      }
      if( cSep=='(' ){
        sqlite3_free(zCreate);
        sqlite3_free(sCtx.z);
        xCloser(sCtx.in);
        utf8_printf(stderr,"%s: empty file\n", sCtx.zFile);
        return 1;
      }
      zCreate = sqlite3_mprintf("%z\n)", zCreate);
      rc = sqlite3_exec(p->db, zCreate, 0, 0, 0);
      sqlite3_free(zCreate);
      if( rc ){
        utf8_printf(stderr, "CREATE TABLE %s(...) failed: %s\n", zTable,
                sqlite3_errmsg(p->db));
        sqlite3_free(sCtx.z);
        xCloser(sCtx.in);
        return 1;
      }
      rc = sqlite3_prepare_v2(p->db, zSql, -1, &pStmt, 0);
    }
    sqlite3_free(zSql);
    if( rc ){
      if (pStmt) sqlite3_finalize(pStmt);
      utf8_printf(stderr,"Error: %s\n", sqlite3_errmsg(p->db));
      xCloser(sCtx.in);
      return 1;
    }
    nCol = sqlite3_column_count(pStmt);
    sqlite3_finalize(pStmt);
    pStmt = 0;
    if( nCol==0 ) return 0; /* no columns, no error */
    zSql = sqlite3_malloc64( nByte*2 + 20 + nCol*2 );
    if( zSql==0 ){
      raw_printf(stderr, "Error: out of memory\n");
      xCloser(sCtx.in);
      return 1;
    }
    sqlite3_snprintf(nByte+20, zSql, "INSERT INTO \"%w\" VALUES(?", zTable);
    j = strlen30(zSql);
    for(i=1; i<nCol; i++){
      zSql[j++] = ',';
      zSql[j++] = '?';
    }
    zSql[j++] = ')';
    zSql[j] = 0;
    rc = sqlite3_prepare_v2(p->db, zSql, -1, &pStmt, 0);
    sqlite3_free(zSql);
    if( rc ){
      utf8_printf(stderr, "Error: %s\n", sqlite3_errmsg(p->db));
      if (pStmt) sqlite3_finalize(pStmt);
      xCloser(sCtx.in);
      return 1;
    }
    needCommit = sqlite3_get_autocommit(p->db);
    if( needCommit ) sqlite3_exec(p->db, "BEGIN", 0, 0, 0);
    do{
      int startLine = sCtx.nLine;
      for(i=0; i<nCol; i++){
        char *z = xRead(&sCtx);
        /*
        ** Did we reach end-of-file before finding any columns?
        ** If so, stop instead of NULL filling the remaining columns.
        */
        if( z==0 && i==0 ) break;
        /*
        ** Did we reach end-of-file OR end-of-line before finding any
        ** columns in ASCII mode?  If so, stop instead of NULL filling
        ** the remaining columns.
        */
        if( p->mode==MODE_Ascii && (z==0 || z[0]==0) && i==0 ) break;
        sqlite3_bind_text(pStmt, i+1, z, -1, SQLITE_TRANSIENT);
        if( i<nCol-1 && sCtx.cTerm!=sCtx.cColSep ){
          utf8_printf(stderr, "%s:%d: expected %d columns but found %d - "
                          "filling the rest with NULL\n",
                          sCtx.zFile, startLine, nCol, i+1);
          i += 2;
          while( i<=nCol ){ sqlite3_bind_null(pStmt, i); i++; }
        }
      }
      if( sCtx.cTerm==sCtx.cColSep ){
        do{
          xRead(&sCtx);
          i++;
        }while( sCtx.cTerm==sCtx.cColSep );
        utf8_printf(stderr, "%s:%d: expected %d columns but found %d - "
                        "extras ignored\n",
                        sCtx.zFile, startLine, nCol, i);
      }
      if( i>=nCol ){
        sqlite3_step(pStmt);
        rc = sqlite3_reset(pStmt);
        if( rc!=SQLITE_OK ){
          utf8_printf(stderr, "%s:%d: INSERT failed: %s\n", sCtx.zFile,
                      startLine, sqlite3_errmsg(p->db));
        }
      }
    }while( sCtx.cTerm!=EOF );

    xCloser(sCtx.in);
    sqlite3_free(sCtx.z);
    sqlite3_finalize(pStmt);
    if( needCommit ) sqlite3_exec(p->db, "COMMIT", 0, 0, 0);
  }else

#ifndef SQLITE_UNTESTABLE
  if( c=='i' && strncmp(azArg[0], "imposter", n)==0 ){
    char *zSql;
    char *zCollist = 0;
    sqlite3_stmt *pStmt;
    int tnum = 0;
    int i;
    if( nArg!=3 ){
      utf8_printf(stderr, "Usage: .imposter INDEX IMPOSTER\n");
      rc = 1;
      goto meta_command_exit;
    }
    open_db(p, 0);
    zSql = sqlite3_mprintf("SELECT rootpage FROM sqlite_master"
                           " WHERE name='%q' AND type='index'", azArg[1]);
    sqlite3_prepare_v2(p->db, zSql, -1, &pStmt, 0);
    sqlite3_free(zSql);
    if( sqlite3_step(pStmt)==SQLITE_ROW ){
      tnum = sqlite3_column_int(pStmt, 0);
    }
    sqlite3_finalize(pStmt);
    if( tnum==0 ){
      utf8_printf(stderr, "no such index: \"%s\"\n", azArg[1]);
      rc = 1;
      goto meta_command_exit;
    }
    zSql = sqlite3_mprintf("PRAGMA index_xinfo='%q'", azArg[1]);
    rc = sqlite3_prepare_v2(p->db, zSql, -1, &pStmt, 0);
    sqlite3_free(zSql);
    i = 0;
    while( sqlite3_step(pStmt)==SQLITE_ROW ){
      char zLabel[20];
      const char *zCol = (const char*)sqlite3_column_text(pStmt,2);
      i++;
      if( zCol==0 ){
        if( sqlite3_column_int(pStmt,1)==-1 ){
          zCol = "_ROWID_";
        }else{
          sqlite3_snprintf(sizeof(zLabel),zLabel,"expr%d",i);
          zCol = zLabel;
        }
      }
      if( zCollist==0 ){
        zCollist = sqlite3_mprintf("\"%w\"", zCol);
      }else{
        zCollist = sqlite3_mprintf("%z,\"%w\"", zCollist, zCol);
      }
    }
    sqlite3_finalize(pStmt);
    zSql = sqlite3_mprintf(
          "CREATE TABLE \"%w\"(%s,PRIMARY KEY(%s))WITHOUT ROWID",
          azArg[2], zCollist, zCollist);
    sqlite3_free(zCollist);
    rc = sqlite3_test_control(SQLITE_TESTCTRL_IMPOSTER, p->db, "main", 1, tnum);
    if( rc==SQLITE_OK ){
      rc = sqlite3_exec(p->db, zSql, 0, 0, 0);
      sqlite3_test_control(SQLITE_TESTCTRL_IMPOSTER, p->db, "main", 0, 0);
      if( rc ){
        utf8_printf(stderr, "Error in [%s]: %s\n", zSql, sqlite3_errmsg(p->db));
      }else{
        utf8_printf(stdout, "%s;\n", zSql);
        raw_printf(stdout,
           "WARNING: writing to an imposter table will corrupt the index!\n"
        );
      }
    }else{
      raw_printf(stderr, "SQLITE_TESTCTRL_IMPOSTER returns %d\n", rc);
      rc = 1;
    }
    sqlite3_free(zSql);
  }else
#endif /* !defined(SQLITE_OMIT_TEST_CONTROL) */

#ifdef SQLITE_ENABLE_IOTRACE
  if( c=='i' && strncmp(azArg[0], "iotrace", n)==0 ){
    SQLITE_API extern void (SQLITE_CDECL *sqlite3IoTrace)(const char*, ...);
    if( iotrace && iotrace!=stdout ) fclose(iotrace);
    iotrace = 0;
    if( nArg<2 ){
      sqlite3IoTrace = 0;
    }else if( strcmp(azArg[1], "-")==0 ){
      sqlite3IoTrace = iotracePrintf;
      iotrace = stdout;
    }else{
      iotrace = fopen(azArg[1], "w");
      if( iotrace==0 ){
        utf8_printf(stderr, "Error: cannot open \"%s\"\n", azArg[1]);
        sqlite3IoTrace = 0;
        rc = 1;
      }else{
        sqlite3IoTrace = iotracePrintf;
      }
    }
  }else
#endif

  if( c=='l' && n>=5 && strncmp(azArg[0], "limits", n)==0 ){
    static const struct {
       const char *zLimitName;   /* Name of a limit */
       int limitCode;            /* Integer code for that limit */
    } aLimit[] = {
      { "length",                SQLITE_LIMIT_LENGTH                    },
      { "sql_length",            SQLITE_LIMIT_SQL_LENGTH                },
      { "column",                SQLITE_LIMIT_COLUMN                    },
      { "expr_depth",            SQLITE_LIMIT_EXPR_DEPTH                },
      { "compound_select",       SQLITE_LIMIT_COMPOUND_SELECT           },
      { "vdbe_op",               SQLITE_LIMIT_VDBE_OP                   },
      { "function_arg",          SQLITE_LIMIT_FUNCTION_ARG              },
      { "attached",              SQLITE_LIMIT_ATTACHED                  },
      { "like_pattern_length",   SQLITE_LIMIT_LIKE_PATTERN_LENGTH       },
      { "variable_number",       SQLITE_LIMIT_VARIABLE_NUMBER           },
      { "trigger_depth",         SQLITE_LIMIT_TRIGGER_DEPTH             },
      { "worker_threads",        SQLITE_LIMIT_WORKER_THREADS            },
    };
    int i, n2;
    open_db(p, 0);
    if( nArg==1 ){
      for(i=0; i<ArraySize(aLimit); i++){
        printf("%20s %d\n", aLimit[i].zLimitName,
               sqlite3_limit(p->db, aLimit[i].limitCode, -1));
      }
    }else if( nArg>3 ){
      raw_printf(stderr, "Usage: .limit NAME ?NEW-VALUE?\n");
      rc = 1;
      goto meta_command_exit;
    }else{
      int iLimit = -1;
      n2 = strlen30(azArg[1]);
      for(i=0; i<ArraySize(aLimit); i++){
        if( sqlite3_strnicmp(aLimit[i].zLimitName, azArg[1], n2)==0 ){
          if( iLimit<0 ){
            iLimit = i;
          }else{
            utf8_printf(stderr, "ambiguous limit: \"%s\"\n", azArg[1]);
            rc = 1;
            goto meta_command_exit;
          }
        }
      }
      if( iLimit<0 ){
        utf8_printf(stderr, "unknown limit: \"%s\"\n"
                        "enter \".limits\" with no arguments for a list.\n",
                         azArg[1]);
        rc = 1;
        goto meta_command_exit;
      }
      if( nArg==3 ){
        sqlite3_limit(p->db, aLimit[iLimit].limitCode,
                      (int)integerValue(azArg[2]));
      }
      printf("%20s %d\n", aLimit[iLimit].zLimitName,
             sqlite3_limit(p->db, aLimit[iLimit].limitCode, -1));
    }
  }else

  if( c=='l' && n>2 && strncmp(azArg[0], "lint", n)==0 ){
    open_db(p, 0);
    lintDotCommand(p, azArg, nArg);
  }else

#ifndef SQLITE_OMIT_LOAD_EXTENSION
  if( c=='l' && strncmp(azArg[0], "load", n)==0 ){
    const char *zFile, *zProc;
    char *zErrMsg = 0;
    if( nArg<2 ){
      raw_printf(stderr, "Usage: .load FILE ?ENTRYPOINT?\n");
      rc = 1;
      goto meta_command_exit;
    }
    zFile = azArg[1];
    zProc = nArg>=3 ? azArg[2] : 0;
    open_db(p, 0);
    rc = sqlite3_load_extension(p->db, zFile, zProc, &zErrMsg);
    if( rc!=SQLITE_OK ){
      utf8_printf(stderr, "Error: %s\n", zErrMsg);
      sqlite3_free(zErrMsg);
      rc = 1;
    }
  }else
#endif

  if( c=='l' && strncmp(azArg[0], "log", n)==0 ){
    if( nArg!=2 ){
      raw_printf(stderr, "Usage: .log FILENAME\n");
      rc = 1;
    }else{
      const char *zFile = azArg[1];
      output_file_close(p->pLog);
      p->pLog = output_file_open(zFile);
    }
  }else

  if( c=='m' && strncmp(azArg[0], "mode", n)==0 ){
    const char *zMode = nArg>=2 ? azArg[1] : "";
    int n2 = (int)strlen(zMode);
    int c2 = zMode[0];
    if( c2=='l' && n2>2 && strncmp(azArg[1],"lines",n2)==0 ){
      p->mode = MODE_Line;
      sqlite3_snprintf(sizeof(p->rowSeparator), p->rowSeparator, SEP_Row);
    }else if( c2=='c' && strncmp(azArg[1],"columns",n2)==0 ){
      p->mode = MODE_Column;
      sqlite3_snprintf(sizeof(p->rowSeparator), p->rowSeparator, SEP_Row);
    }else if( c2=='l' && n2>2 && strncmp(azArg[1],"list",n2)==0 ){
      p->mode = MODE_List;
      sqlite3_snprintf(sizeof(p->colSeparator), p->colSeparator, SEP_Column);
      sqlite3_snprintf(sizeof(p->rowSeparator), p->rowSeparator, SEP_Row);
    }else if( c2=='h' && strncmp(azArg[1],"html",n2)==0 ){
      p->mode = MODE_Html;
    }else if( c2=='t' && strncmp(azArg[1],"tcl",n2)==0 ){
      p->mode = MODE_Tcl;
      sqlite3_snprintf(sizeof(p->colSeparator), p->colSeparator, SEP_Space);
      sqlite3_snprintf(sizeof(p->rowSeparator), p->rowSeparator, SEP_Row);
    }else if( c2=='c' && strncmp(azArg[1],"csv",n2)==0 ){
      p->mode = MODE_Csv;
      sqlite3_snprintf(sizeof(p->colSeparator), p->colSeparator, SEP_Comma);
      sqlite3_snprintf(sizeof(p->rowSeparator), p->rowSeparator, SEP_CrLf);
    }else if( c2=='t' && strncmp(azArg[1],"tabs",n2)==0 ){
      p->mode = MODE_List;
      sqlite3_snprintf(sizeof(p->colSeparator), p->colSeparator, SEP_Tab);
    }else if( c2=='i' && strncmp(azArg[1],"insert",n2)==0 ){
      p->mode = MODE_Insert;
      set_table_name(p, nArg>=3 ? azArg[2] : "table");
    }else if( c2=='q' && strncmp(azArg[1],"quote",n2)==0 ){
      p->mode = MODE_Quote;
    }else if( c2=='a' && strncmp(azArg[1],"ascii",n2)==0 ){
      p->mode = MODE_Ascii;
      sqlite3_snprintf(sizeof(p->colSeparator), p->colSeparator, SEP_Unit);
      sqlite3_snprintf(sizeof(p->rowSeparator), p->rowSeparator, SEP_Record);
    }else {
      raw_printf(stderr, "Error: mode should be one of: "
         "ascii column csv html insert line list quote tabs tcl\n");
      rc = 1;
    }
    p->cMode = p->mode;
  }else

  if( c=='n' && strncmp(azArg[0], "nullvalue", n)==0 ){
    if( nArg==2 ){
      sqlite3_snprintf(sizeof(p->nullValue), p->nullValue,
                       "%.*s", (int)ArraySize(p->nullValue)-1, azArg[1]);
    }else{
      raw_printf(stderr, "Usage: .nullvalue STRING\n");
      rc = 1;
    }
  }else

  if( c=='o' && strncmp(azArg[0], "open", n)==0 && n>=2 ){
    char *zNewFilename;  /* Name of the database file to open */
    int iName = 1;       /* Index in azArg[] of the filename */
    int newFlag = 0;     /* True to delete file before opening */
    /* Close the existing database */
    session_close_all(p);
    sqlite3_close(p->db);
    p->db = 0;
    p->zDbFilename = 0;
    sqlite3_free(p->zFreeOnClose);
    p->zFreeOnClose = 0;
    /* Check for command-line arguments */
    for(iName=1; iName<nArg && azArg[iName][0]=='-'; iName++){
      const char *z = azArg[iName];
      if( optionMatch(z,"new") ){
        newFlag = 1;
      }else if( z[0]=='-' ){
        utf8_printf(stderr, "unknown option: %s\n", z);
        rc = 1;
        goto meta_command_exit;
      }
    }
    /* If a filename is specified, try to open it first */
    zNewFilename = nArg>iName ? sqlite3_mprintf("%s", azArg[iName]) : 0;
    if( zNewFilename ){
      if( newFlag ) shellDeleteFile(zNewFilename);
      p->zDbFilename = zNewFilename;
      open_db(p, 1);
      if( p->db==0 ){
        utf8_printf(stderr, "Error: cannot open '%s'\n", zNewFilename);
        sqlite3_free(zNewFilename);
      }else{
        p->zFreeOnClose = zNewFilename;
      }
    }
    if( p->db==0 ){
      /* As a fall-back open a TEMP database */
      p->zDbFilename = 0;
      open_db(p, 0);
    }
  }else

  if( c=='o'
   && (strncmp(azArg[0], "output", n)==0 || strncmp(azArg[0], "once", n)==0)
  ){
    const char *zFile = nArg>=2 ? azArg[1] : "stdout";
    if( nArg>2 ){
      utf8_printf(stderr, "Usage: .%s FILE\n", azArg[0]);
      rc = 1;
      goto meta_command_exit;
    }
    if( n>1 && strncmp(azArg[0], "once", n)==0 ){
      if( nArg<2 ){
        raw_printf(stderr, "Usage: .once FILE\n");
        rc = 1;
        goto meta_command_exit;
      }
      p->outCount = 2;
    }else{
      p->outCount = 0;
    }
    output_reset(p);
    if( zFile[0]=='|' ){
#ifdef SQLITE_OMIT_POPEN
      raw_printf(stderr, "Error: pipes are not supported in this OS\n");
      rc = 1;
      p->out = stdout;
#else
      p->out = popen(zFile + 1, "w");
      if( p->out==0 ){
        utf8_printf(stderr,"Error: cannot open pipe \"%s\"\n", zFile + 1);
        p->out = stdout;
        rc = 1;
      }else{
        sqlite3_snprintf(sizeof(p->outfile), p->outfile, "%s", zFile);
      }
#endif
    }else{
      p->out = output_file_open(zFile);
      if( p->out==0 ){
        if( strcmp(zFile,"off")!=0 ){
          utf8_printf(stderr,"Error: cannot write to \"%s\"\n", zFile);
        }
        p->out = stdout;
        rc = 1;
      } else {
        sqlite3_snprintf(sizeof(p->outfile), p->outfile, "%s", zFile);
      }
    }
  }else

  if( c=='p' && n>=3 && strncmp(azArg[0], "print", n)==0 ){
    int i;
    for(i=1; i<nArg; i++){
      if( i>1 ) raw_printf(p->out, " ");
      utf8_printf(p->out, "%s", azArg[i]);
    }
    raw_printf(p->out, "\n");
  }else

  if( c=='p' && strncmp(azArg[0], "prompt", n)==0 ){
    if( nArg >= 2) {
      strncpy(mainPrompt,azArg[1],(int)ArraySize(mainPrompt)-1);
    }
    if( nArg >= 3) {
      strncpy(continuePrompt,azArg[2],(int)ArraySize(continuePrompt)-1);
    }
  }else

  if( c=='q' && strncmp(azArg[0], "quit", n)==0 ){
    rc = 2;
  }else

  if( c=='r' && n>=3 && strncmp(azArg[0], "read", n)==0 ){
    FILE *alt;
    if( nArg!=2 ){
      raw_printf(stderr, "Usage: .read FILE\n");
      rc = 1;
      goto meta_command_exit;
    }
    alt = fopen(azArg[1], "rb");
    if( alt==0 ){
      utf8_printf(stderr,"Error: cannot open \"%s\"\n", azArg[1]);
      rc = 1;
    }else{
      rc = process_input(p, alt);
      fclose(alt);
    }
  }else

  if( c=='r' && n>=3 && strncmp(azArg[0], "restore", n)==0 ){
    const char *zSrcFile;
    const char *zDb;
    sqlite3 *pSrc;
    sqlite3_backup *pBackup;
    int nTimeout = 0;

    if( nArg==2 ){
      zSrcFile = azArg[1];
      zDb = "main";
    }else if( nArg==3 ){
      zSrcFile = azArg[2];
      zDb = azArg[1];
    }else{
      raw_printf(stderr, "Usage: .restore ?DB? FILE\n");
      rc = 1;
      goto meta_command_exit;
    }
    rc = sqlite3_open(zSrcFile, &pSrc);
    if( rc!=SQLITE_OK ){
      utf8_printf(stderr, "Error: cannot open \"%s\"\n", zSrcFile);
      sqlite3_close(pSrc);
      return 1;
    }
    open_db(p, 0);
    pBackup = sqlite3_backup_init(p->db, zDb, pSrc, "main");
    if( pBackup==0 ){
      utf8_printf(stderr, "Error: %s\n", sqlite3_errmsg(p->db));
      sqlite3_close(pSrc);
      return 1;
    }
    while( (rc = sqlite3_backup_step(pBackup,100))==SQLITE_OK
          || rc==SQLITE_BUSY  ){
      if( rc==SQLITE_BUSY ){
        if( nTimeout++ >= 3 ) break;
        sqlite3_sleep(100);
      }
    }
    sqlite3_backup_finish(pBackup);
    if( rc==SQLITE_DONE ){
      rc = 0;
    }else if( rc==SQLITE_BUSY || rc==SQLITE_LOCKED ){
      raw_printf(stderr, "Error: source database is busy\n");
      rc = 1;
    }else{
      utf8_printf(stderr, "Error: %s\n", sqlite3_errmsg(p->db));
      rc = 1;
    }
    sqlite3_close(pSrc);
  }else


  if( c=='s' && strncmp(azArg[0], "scanstats", n)==0 ){
    if( nArg==2 ){
      p->scanstatsOn = booleanValue(azArg[1]);
#ifndef SQLITE_ENABLE_STMT_SCANSTATUS
      raw_printf(stderr, "Warning: .scanstats not available in this build.\n");
#endif
    }else{
      raw_printf(stderr, "Usage: .scanstats on|off\n");
      rc = 1;
    }
  }else

  if( c=='s' && strncmp(azArg[0], "schema", n)==0 ){
    ShellState data;
    char *zErrMsg = 0;
    open_db(p, 0);
    memcpy(&data, p, sizeof(data));
    data.showHeader = 0;
    data.cMode = data.mode = MODE_Semi;
    if( nArg>=2 && optionMatch(azArg[1], "indent") ){
      data.cMode = data.mode = MODE_Pretty;
      nArg--;
      if( nArg==2 ) azArg[1] = azArg[2];
    }
    if( nArg==2 && azArg[1][0]!='-' ){
      int i;
      for(i=0; azArg[1][i]; i++) azArg[1][i] = ToLower(azArg[1][i]);
      if( strcmp(azArg[1],"sqlite_master")==0 ){
        char *new_argv[2], *new_colv[2];
        new_argv[0] = "CREATE TABLE sqlite_master (\n"
                      "  type text,\n"
                      "  name text,\n"
                      "  tbl_name text,\n"
                      "  rootpage integer,\n"
                      "  sql text\n"
                      ")";
        new_argv[1] = 0;
        new_colv[0] = "sql";
        new_colv[1] = 0;
        callback(&data, 1, new_argv, new_colv);
        rc = SQLITE_OK;
      }else if( strcmp(azArg[1],"sqlite_temp_master")==0 ){
        char *new_argv[2], *new_colv[2];
        new_argv[0] = "CREATE TEMP TABLE sqlite_temp_master (\n"
                      "  type text,\n"
                      "  name text,\n"
                      "  tbl_name text,\n"
                      "  rootpage integer,\n"
                      "  sql text\n"
                      ")";
        new_argv[1] = 0;
        new_colv[0] = "sql";
        new_colv[1] = 0;
        callback(&data, 1, new_argv, new_colv);
        rc = SQLITE_OK;
      }else{
        char *zSql;
        zSql = sqlite3_mprintf(
          "SELECT sql FROM "
          "  (SELECT sql sql, type type, tbl_name tbl_name, name name, rowid x"
          "     FROM sqlite_master UNION ALL"
          "   SELECT sql, type, tbl_name, name, rowid FROM sqlite_temp_master) "
          "WHERE lower(tbl_name) LIKE %Q"
          "  AND type!='meta' AND sql NOTNULL "
          "ORDER BY rowid", azArg[1]);
        rc = sqlite3_exec(p->db, zSql, callback, &data, &zErrMsg);
        sqlite3_free(zSql);
      }
    }else if( nArg==1 ){
      rc = sqlite3_exec(p->db,
         "SELECT sql FROM "
         "  (SELECT sql sql, type type, tbl_name tbl_name, name name, rowid x"
         "     FROM sqlite_master UNION ALL"
         "   SELECT sql, type, tbl_name, name, rowid FROM sqlite_temp_master) "
         "WHERE type!='meta' AND sql NOTNULL AND name NOT LIKE 'sqlite_%' "
         "ORDER BY rowid",
         callback, &data, &zErrMsg
      );
    }else{
      raw_printf(stderr, "Usage: .schema ?--indent? ?LIKE-PATTERN?\n");
      rc = 1;
      goto meta_command_exit;
    }
    if( zErrMsg ){
      utf8_printf(stderr,"Error: %s\n", zErrMsg);
      sqlite3_free(zErrMsg);
      rc = 1;
    }else if( rc != SQLITE_OK ){
      raw_printf(stderr,"Error: querying schema information\n");
      rc = 1;
    }else{
      rc = 0;
    }
  }else

#if defined(SQLITE_DEBUG) && defined(SQLITE_ENABLE_SELECTTRACE)
  if( c=='s' && n==11 && strncmp(azArg[0], "selecttrace", n)==0 ){
    sqlite3SelectTrace = (int)integerValue(azArg[1]);
  }else
#endif

#if defined(SQLITE_ENABLE_SESSION)
  if( c=='s' && strncmp(azArg[0],"session",n)==0 && n>=3 ){
    OpenSession *pSession = &p->aSession[0];
    char **azCmd = &azArg[1];
    int iSes = 0;
    int nCmd = nArg - 1;
    int i;
    if( nArg<=1 ) goto session_syntax_error;
    open_db(p, 0);
    if( nArg>=3 ){
      for(iSes=0; iSes<p->nSession; iSes++){
        if( strcmp(p->aSession[iSes].zName, azArg[1])==0 ) break;
      }
      if( iSes<p->nSession ){
        pSession = &p->aSession[iSes];
        azCmd++;
        nCmd--;
      }else{
        pSession = &p->aSession[0];
        iSes = 0;
      }
    }

    /* .session attach TABLE
    ** Invoke the sqlite3session_attach() interface to attach a particular
    ** table so that it is never filtered.
    */
    if( strcmp(azCmd[0],"attach")==0 ){
      if( nCmd!=2 ) goto session_syntax_error;
      if( pSession->p==0 ){
        session_not_open:
        raw_printf(stderr, "ERROR: No sessions are open\n");
      }else{
        rc = sqlite3session_attach(pSession->p, azCmd[1]);
        if( rc ){
          raw_printf(stderr, "ERROR: sqlite3session_attach() returns %d\n", rc);
          rc = 0;
        }
      }
    }else

    /* .session changeset FILE
    ** .session patchset FILE
    ** Write a changeset or patchset into a file.  The file is overwritten.
    */
    if( strcmp(azCmd[0],"changeset")==0 || strcmp(azCmd[0],"patchset")==0 ){
      FILE *out = 0;
      if( nCmd!=2 ) goto session_syntax_error;
      if( pSession->p==0 ) goto session_not_open;
      out = fopen(azCmd[1], "wb");
      if( out==0 ){
        utf8_printf(stderr, "ERROR: cannot open \"%s\" for writing\n", azCmd[1]);
      }else{
        int szChng;
        void *pChng;
        if( azCmd[0][0]=='c' ){
          rc = sqlite3session_changeset(pSession->p, &szChng, &pChng);
        }else{
          rc = sqlite3session_patchset(pSession->p, &szChng, &pChng);
        }
        if( rc ){
          printf("Error: error code %d\n", rc);
          rc = 0;
        }
        if( pChng
          && fwrite(pChng, szChng, 1, out)!=1 ){
          raw_printf(stderr, "ERROR: Failed to write entire %d-byte output\n",
                  szChng);
        }
        sqlite3_free(pChng);
        fclose(out);
      }
    }else

    /* .session close
    ** Close the identified session
    */
    if( strcmp(azCmd[0], "close")==0 ){
      if( nCmd!=1 ) goto session_syntax_error;
      if( p->nSession ){
        session_close(pSession);
        p->aSession[iSes] = p->aSession[--p->nSession];
      }
    }else

    /* .session enable ?BOOLEAN?
    ** Query or set the enable flag
    */
    if( strcmp(azCmd[0], "enable")==0 ){
      int ii;
      if( nCmd>2 ) goto session_syntax_error;
      ii = nCmd==1 ? -1 : booleanValue(azCmd[1]);
      if( p->nSession ){
        ii = sqlite3session_enable(pSession->p, ii);
        utf8_printf(p->out, "session %s enable flag = %d\n",
                    pSession->zName, ii);
      }
    }else

    /* .session filter GLOB ....
    ** Set a list of GLOB patterns of table names to be excluded.
    */
    if( strcmp(azCmd[0], "filter")==0 ){
      int ii, nByte;
      if( nCmd<2 ) goto session_syntax_error;
      if( p->nSession ){
        for(ii=0; ii<pSession->nFilter; ii++){
          sqlite3_free(pSession->azFilter[ii]);
        }
        sqlite3_free(pSession->azFilter);
        nByte = sizeof(pSession->azFilter[0])*(nCmd-1);
        pSession->azFilter = sqlite3_malloc( nByte );
        if( pSession->azFilter==0 ){
          raw_printf(stderr, "Error: out or memory\n");
          exit(1);
        }
        for(ii=1; ii<nCmd; ii++){
          pSession->azFilter[ii-1] = sqlite3_mprintf("%s", azCmd[ii]);
        }
        pSession->nFilter = ii-1;
      }
    }else

    /* .session indirect ?BOOLEAN?
    ** Query or set the indirect flag
    */
    if( strcmp(azCmd[0], "indirect")==0 ){
      int ii;
      if( nCmd>2 ) goto session_syntax_error;
      ii = nCmd==1 ? -1 : booleanValue(azCmd[1]);
      if( p->nSession ){
        ii = sqlite3session_indirect(pSession->p, ii);
        utf8_printf(p->out, "session %s indirect flag = %d\n",
                    pSession->zName, ii);
      }
    }else

    /* .session isempty
    ** Determine if the session is empty
    */
    if( strcmp(azCmd[0], "isempty")==0 ){
      int ii;
      if( nCmd!=1 ) goto session_syntax_error;
      if( p->nSession ){
        ii = sqlite3session_isempty(pSession->p);
        utf8_printf(p->out, "session %s isempty flag = %d\n",
                    pSession->zName, ii);
      }
    }else

    /* .session list
    ** List all currently open sessions
    */
    if( strcmp(azCmd[0],"list")==0 ){
      for(i=0; i<p->nSession; i++){
        utf8_printf(p->out, "%d %s\n", i, p->aSession[i].zName);
      }
    }else

    /* .session open DB NAME
    ** Open a new session called NAME on the attached database DB.
    ** DB is normally "main".
    */
    if( strcmp(azCmd[0],"open")==0 ){
      char *zName;
      if( nCmd!=3 ) goto session_syntax_error;
      zName = azCmd[2];
      if( zName[0]==0 ) goto session_syntax_error;
      for(i=0; i<p->nSession; i++){
        if( strcmp(p->aSession[i].zName,zName)==0 ){
          utf8_printf(stderr, "Session \"%s\" already exists\n", zName);
          goto meta_command_exit;
        }
      }
      if( p->nSession>=ArraySize(p->aSession) ){
        raw_printf(stderr, "Maximum of %d sessions\n", ArraySize(p->aSession));
        goto meta_command_exit;
      }
      pSession = &p->aSession[p->nSession];
      rc = sqlite3session_create(p->db, azCmd[1], &pSession->p);
      if( rc ){
        raw_printf(stderr, "Cannot open session: error code=%d\n", rc);
        rc = 0;
        goto meta_command_exit;
      }
      pSession->nFilter = 0;
      sqlite3session_table_filter(pSession->p, session_filter, pSession);
      p->nSession++;
      pSession->zName = sqlite3_mprintf("%s", zName);
    }else
    /* If no command name matches, show a syntax error */
    session_syntax_error:
    session_help(p);
  }else
#endif

#ifdef SQLITE_DEBUG
  /* Undocumented commands for internal testing.  Subject to change
  ** without notice. */
  if( c=='s' && n>=10 && strncmp(azArg[0], "selftest-", 9)==0 ){
    if( strncmp(azArg[0]+9, "boolean", n-9)==0 ){
      int i, v;
      for(i=1; i<nArg; i++){
        v = booleanValue(azArg[i]);
        utf8_printf(p->out, "%s: %d 0x%x\n", azArg[i], v, v);
      }
    }
    if( strncmp(azArg[0]+9, "integer", n-9)==0 ){
      int i; sqlite3_int64 v;
      for(i=1; i<nArg; i++){
        char zBuf[200];
        v = integerValue(azArg[i]);
        sqlite3_snprintf(sizeof(zBuf),zBuf,"%s: %lld 0x%llx\n", azArg[i],v,v);
        utf8_printf(p->out, "%s", zBuf);
      }
    }
  }else
#endif

  if( c=='s' && n>=4 && strncmp(azArg[0],"selftest",n)==0 ){
    int bIsInit = 0;         /* True to initialize the SELFTEST table */
    int bVerbose = 0;        /* Verbose output */
    int bSelftestExists;     /* True if SELFTEST already exists */
    char **azTest = 0;       /* Content of the SELFTEST table */
    int nRow = 0;            /* Number of rows in the SELFTEST table */
    int nCol = 4;            /* Number of columns in the SELFTEST table */
    int i;                   /* Loop counter */
    int nTest = 0;           /* Number of tests runs */
    int nErr = 0;            /* Number of errors seen */
    ShellText str;           /* Answer for a query */
    static char *azDefaultTest[] = {
       0, 0, 0, 0,
       "0", "memo", "Missing SELFTEST table - default checks only", "",
       "1", "run", "PRAGMA integrity_check", "ok"
    };
    static const int nDefaultRow = 2;

    open_db(p,0);
    for(i=1; i<nArg; i++){
      const char *z = azArg[i];
      if( z[0]=='-' && z[1]=='-' ) z++;
      if( strcmp(z,"-init")==0 ){
        bIsInit = 1;
      }else
      if( strcmp(z,"-v")==0 ){
        bVerbose++;
      }else
      {
        utf8_printf(stderr, "Unknown option \"%s\" on \"%s\"\n",
                    azArg[i], azArg[0]);
        raw_printf(stderr, "Should be one of: --init -v\n");
        rc = 1;
        goto meta_command_exit;
      }
    }
    if( sqlite3_table_column_metadata(p->db,"main","selftest",0,0,0,0,0,0)
           != SQLITE_OK ){
      bSelftestExists = 0;
    }else{
      bSelftestExists = 1;
    }
    if( bIsInit ){
      createSelftestTable(p);
      bSelftestExists = 1;
    }
    if( bSelftestExists ){
      rc = sqlite3_get_table(p->db, 
          "SELECT tno,op,cmd,ans FROM selftest ORDER BY tno",
          &azTest, &nRow, &nCol, 0);
      if( rc ){
        raw_printf(stderr, "Error querying the selftest table\n");
        rc = 1;
        sqlite3_free_table(azTest);
        goto meta_command_exit;
      }else if( nRow==0 ){
        sqlite3_free_table(azTest);
        azTest = azDefaultTest;
        nRow = nDefaultRow;
      }
    }else{
      azTest = azDefaultTest;
      nRow = nDefaultRow;
    }
    initText(&str);
    appendText(&str, "x", 0);
    for(i=1; i<=nRow; i++){
      int tno = atoi(azTest[i*nCol]);
      const char *zOp = azTest[i*nCol+1];
      const char *zSql = azTest[i*nCol+2];
      const char *zAns = azTest[i*nCol+3];
  
      if( bVerbose>0 ){
        char *zQuote = sqlite3_mprintf("%q", zSql);
        printf("%d: %s %s\n", tno, zOp, zSql);
        sqlite3_free(zQuote);
      }
      if( strcmp(zOp,"memo")==0 ){
        utf8_printf(p->out, "%s\n", zSql);
      }else
      if( strcmp(zOp,"run")==0 ){
        char *zErrMsg = 0;
        str.n = 0;
        str.z[0] = 0;
        rc = sqlite3_exec(p->db, zSql, captureOutputCallback, &str, &zErrMsg);
        nTest++;
        if( bVerbose ){
          utf8_printf(p->out, "Result: %s\n", str.z);
        }
        if( rc || zErrMsg ){
          nErr++;
          rc = 1;
          utf8_printf(p->out, "%d: error-code-%d: %s\n", tno, rc, zErrMsg);
          sqlite3_free(zErrMsg);
        }else if( strcmp(zAns,str.z)!=0 ){
          nErr++;
          rc = 1;
          utf8_printf(p->out, "%d: Expected: [%s]\n", tno, zAns);
          utf8_printf(p->out, "%d:      Got: [%s]\n", tno, str.z);
        }
      }else
      {
        utf8_printf(stderr,
          "Unknown operation \"%s\" on selftest line %d\n", zOp, tno);
        rc = 1;
        break;
      }
    }
    freeText(&str);
    if( azTest!=azDefaultTest ) sqlite3_free_table(azTest);
    utf8_printf(p->out, "%d errors out of %d tests\n", nErr, nTest);
  }else

  if( c=='s' && strncmp(azArg[0], "separator", n)==0 ){
    if( nArg<2 || nArg>3 ){
      raw_printf(stderr, "Usage: .separator COL ?ROW?\n");
      rc = 1;
    }
    if( nArg>=2 ){
      sqlite3_snprintf(sizeof(p->colSeparator), p->colSeparator,
                       "%.*s", (int)ArraySize(p->colSeparator)-1, azArg[1]);
    }
    if( nArg>=3 ){
      sqlite3_snprintf(sizeof(p->rowSeparator), p->rowSeparator,
                       "%.*s", (int)ArraySize(p->rowSeparator)-1, azArg[2]);
    }
  }else

  if( c=='s' && n>=4 && strncmp(azArg[0],"sha3sum",n)==0 ){
    const char *zLike = 0;   /* Which table to checksum. 0 means everything */
    int i;                   /* Loop counter */
    int bSchema = 0;         /* Also hash the schema */
    int bSeparate = 0;       /* Hash each table separately */
    int iSize = 224;         /* Hash algorithm to use */
    int bDebug = 0;          /* Only show the query that would have run */
    sqlite3_stmt *pStmt;     /* For querying tables names */
    char *zSql;              /* SQL to be run */
    char *zSep;              /* Separator */
    ShellText sSql;          /* Complete SQL for the query to run the hash */
    ShellText sQuery;        /* Set of queries used to read all content */
    open_db(p, 0);
    for(i=1; i<nArg; i++){
      const char *z = azArg[i];
      if( z[0]=='-' ){
        z++;
        if( z[0]=='-' ) z++;
        if( strcmp(z,"schema")==0 ){
          bSchema = 1;
        }else
        if( strcmp(z,"sha3-224")==0 || strcmp(z,"sha3-256")==0 
         || strcmp(z,"sha3-384")==0 || strcmp(z,"sha3-512")==0 
        ){
          iSize = atoi(&z[5]);
        }else
        if( strcmp(z,"debug")==0 ){
          bDebug = 1;
        }else
        {
          utf8_printf(stderr, "Unknown option \"%s\" on \"%s\"\n",
                      azArg[i], azArg[0]);
          raw_printf(stderr, "Should be one of: --schema"
                             " --sha3-224 --sha3-255 --sha3-384 --sha3-512\n");
          rc = 1;
          goto meta_command_exit;
        }
      }else if( zLike ){
        raw_printf(stderr, "Usage: .sha3sum ?OPTIONS? ?LIKE-PATTERN?\n");
        rc = 1;
        goto meta_command_exit;
      }else{
        zLike = z;
        bSeparate = 1;
        if( sqlite3_strlike("sqlite_%", zLike, 0)==0 ) bSchema = 1;
      }
    }
    if( bSchema ){
      zSql = "SELECT lower(name) FROM sqlite_master"
             " WHERE type='table' AND coalesce(rootpage,0)>1"
             " UNION ALL SELECT 'sqlite_master'"
             " ORDER BY 1 collate nocase";
    }else{
      zSql = "SELECT lower(name) FROM sqlite_master"
             " WHERE type='table' AND coalesce(rootpage,0)>1"
             " AND name NOT LIKE 'sqlite_%'"
             " ORDER BY 1 collate nocase";
    }
    sqlite3_prepare_v2(p->db, zSql, -1, &pStmt, 0);
    initText(&sQuery);
    initText(&sSql);
    appendText(&sSql, "WITH [sha3sum$query](a,b) AS(",0);
    zSep = "VALUES(";
    while( SQLITE_ROW==sqlite3_step(pStmt) ){
      const char *zTab = (const char*)sqlite3_column_text(pStmt,0);
      if( zLike && sqlite3_strlike(zLike, zTab, 0)!=0 ) continue;
      if( strncmp(zTab, "sqlite_",7)!=0 ){
        appendText(&sQuery,"SELECT * FROM ", 0);
        appendText(&sQuery,zTab,'"');
        appendText(&sQuery," NOT INDEXED;", 0);
      }else if( strcmp(zTab, "sqlite_master")==0 ){
        appendText(&sQuery,"SELECT type,name,tbl_name,sql FROM sqlite_master"
                           " ORDER BY name;", 0);
      }else if( strcmp(zTab, "sqlite_sequence")==0 ){
        appendText(&sQuery,"SELECT name,seq FROM sqlite_sequence"
                           " ORDER BY name;", 0);
      }else if( strcmp(zTab, "sqlite_stat1")==0 ){
        appendText(&sQuery,"SELECT tbl,idx,stat FROM sqlite_stat1"
                           " ORDER BY tbl,idx;", 0);
      }else if( strcmp(zTab, "sqlite_stat3")==0
             || strcmp(zTab, "sqlite_stat4")==0 ){
        appendText(&sQuery, "SELECT * FROM ", 0);
        appendText(&sQuery, zTab, 0);
        appendText(&sQuery, " ORDER BY tbl, idx, rowid;\n", 0);
      }
      appendText(&sSql, zSep, 0);
      appendText(&sSql, sQuery.z, '\'');
      sQuery.n = 0;
      appendText(&sSql, ",", 0);
      appendText(&sSql, zTab, '\'');
      zSep = "),(";
    }
    sqlite3_finalize(pStmt);
    if( bSeparate ){
      zSql = sqlite3_mprintf(
          "%s))"
          " SELECT lower(hex(sha3_query(a,%d))) AS hash, b AS label"
          "   FROM [sha3sum$query]",
          sSql.z, iSize);
    }else{
      zSql = sqlite3_mprintf(
          "%s))"
          " SELECT lower(hex(sha3_query(group_concat(a,''),%d))) AS hash"
          "   FROM [sha3sum$query]",
          sSql.z, iSize);
    }
    freeText(&sQuery);
    freeText(&sSql);
    if( bDebug ){
      utf8_printf(p->out, "%s\n", zSql);
    }else{
      shell_exec(p->db, zSql, shell_callback, p, 0);
    }
    sqlite3_free(zSql);
  }else

  if( c=='s'
   && (strncmp(azArg[0], "shell", n)==0 || strncmp(azArg[0],"system",n)==0)
  ){
    char *zCmd;
    int i, x;
    if( nArg<2 ){
      raw_printf(stderr, "Usage: .system COMMAND\n");
      rc = 1;
      goto meta_command_exit;
    }
    zCmd = sqlite3_mprintf(strchr(azArg[1],' ')==0?"%s":"\"%s\"", azArg[1]);
    for(i=2; i<nArg; i++){
      zCmd = sqlite3_mprintf(strchr(azArg[i],' ')==0?"%z %s":"%z \"%s\"",
                             zCmd, azArg[i]);
    }
    x = system(zCmd);
    sqlite3_free(zCmd);
    if( x ) raw_printf(stderr, "System command returns %d\n", x);
  }else

  if( c=='s' && strncmp(azArg[0], "show", n)==0 ){
    static const char *azBool[] = { "off", "on", "full", "unk" };
    int i;
    if( nArg!=1 ){
      raw_printf(stderr, "Usage: .show\n");
      rc = 1;
      goto meta_command_exit;
    }
    utf8_printf(p->out, "%12.12s: %s\n","echo",
                                  azBool[ShellHasFlag(p, SHFLG_Echo)]);
    utf8_printf(p->out, "%12.12s: %s\n","eqp", azBool[p->autoEQP&3]);
    utf8_printf(p->out, "%12.12s: %s\n","explain",
         p->mode==MODE_Explain ? "on" : p->autoExplain ? "auto" : "off");
    utf8_printf(p->out,"%12.12s: %s\n","headers", azBool[p->showHeader!=0]);
    utf8_printf(p->out, "%12.12s: %s\n","mode", modeDescr[p->mode]);
    utf8_printf(p->out, "%12.12s: ", "nullvalue");
      output_c_string(p->out, p->nullValue);
      raw_printf(p->out, "\n");
    utf8_printf(p->out,"%12.12s: %s\n","output",
            strlen30(p->outfile) ? p->outfile : "stdout");
    utf8_printf(p->out,"%12.12s: ", "colseparator");
      output_c_string(p->out, p->colSeparator);
      raw_printf(p->out, "\n");
    utf8_printf(p->out,"%12.12s: ", "rowseparator");
      output_c_string(p->out, p->rowSeparator);
      raw_printf(p->out, "\n");
    utf8_printf(p->out, "%12.12s: %s\n","stats", azBool[p->statsOn!=0]);
    utf8_printf(p->out, "%12.12s: ", "width");
    for (i=0;i<(int)ArraySize(p->colWidth) && p->colWidth[i] != 0;i++) {
      raw_printf(p->out, "%d ", p->colWidth[i]);
    }
    raw_printf(p->out, "\n");
    utf8_printf(p->out, "%12.12s: %s\n", "filename",
                p->zDbFilename ? p->zDbFilename : "");
  }else

  if( c=='s' && strncmp(azArg[0], "stats", n)==0 ){
    if( nArg==2 ){
      p->statsOn = booleanValue(azArg[1]);
    }else if( nArg==1 ){
      display_stats(p->db, p, 0);
    }else{
      raw_printf(stderr, "Usage: .stats ?on|off?\n");
      rc = 1;
    }
  }else

  if( (c=='t' && n>1 && strncmp(azArg[0], "tables", n)==0)
   || (c=='i' && (strncmp(azArg[0], "indices", n)==0
                 || strncmp(azArg[0], "indexes", n)==0) )
  ){
    sqlite3_stmt *pStmt;
    char **azResult;
    int nRow, nAlloc;
    char *zSql = 0;
    int ii;
    open_db(p, 0);
    rc = sqlite3_prepare_v2(p->db, "PRAGMA database_list", -1, &pStmt, 0);
    if( rc ) return shellDatabaseError(p->db);

    /* Create an SQL statement to query for the list of tables in the
    ** main and all attached databases where the table name matches the
    ** LIKE pattern bound to variable "?1". */
    if( c=='t' ){
      zSql = sqlite3_mprintf(
          "SELECT name FROM sqlite_master"
          " WHERE type IN ('table','view')"
          "   AND name NOT LIKE 'sqlite_%%'"
          "   AND name LIKE ?1");
    }else if( nArg>2 ){
      /* It is an historical accident that the .indexes command shows an error
      ** when called with the wrong number of arguments whereas the .tables
      ** command does not. */
      raw_printf(stderr, "Usage: .indexes ?LIKE-PATTERN?\n");
      rc = 1;
      goto meta_command_exit;
    }else{
      zSql = sqlite3_mprintf(
          "SELECT name FROM sqlite_master"
          " WHERE type='index'"
          "   AND tbl_name LIKE ?1");
    }
    for(ii=0; zSql && sqlite3_step(pStmt)==SQLITE_ROW; ii++){
      const char *zDbName = (const char*)sqlite3_column_text(pStmt, 1);
      if( zDbName==0 || ii==0 ) continue;
      if( c=='t' ){
        zSql = sqlite3_mprintf(
                 "%z UNION ALL "
                 "SELECT '%q.' || name FROM \"%w\".sqlite_master"
                 " WHERE type IN ('table','view')"
                 "   AND name NOT LIKE 'sqlite_%%'"
                 "   AND name LIKE ?1", zSql, zDbName, zDbName);
      }else{
        zSql = sqlite3_mprintf(
                 "%z UNION ALL "
                 "SELECT '%q.' || name FROM \"%w\".sqlite_master"
                 " WHERE type='index'"
                 "   AND tbl_name LIKE ?1", zSql, zDbName, zDbName);
      }
    }
    rc = sqlite3_finalize(pStmt);
    if( zSql && rc==SQLITE_OK ){
      zSql = sqlite3_mprintf("%z ORDER BY 1", zSql);
      if( zSql ) rc = sqlite3_prepare_v2(p->db, zSql, -1, &pStmt, 0);
    }
    sqlite3_free(zSql);
    if( !zSql ) return shellNomemError();
    if( rc ) return shellDatabaseError(p->db);

    /* Run the SQL statement prepared by the above block. Store the results
    ** as an array of nul-terminated strings in azResult[].  */
    nRow = nAlloc = 0;
    azResult = 0;
    if( nArg>1 ){
      sqlite3_bind_text(pStmt, 1, azArg[1], -1, SQLITE_TRANSIENT);
    }else{
      sqlite3_bind_text(pStmt, 1, "%", -1, SQLITE_STATIC);
    }
    while( sqlite3_step(pStmt)==SQLITE_ROW ){
      if( nRow>=nAlloc ){
        char **azNew;
        int n2 = nAlloc*2 + 10;
        azNew = sqlite3_realloc64(azResult, sizeof(azResult[0])*n2);
        if( azNew==0 ){
          rc = shellNomemError();
          break;
        }
        nAlloc = n2;
        azResult = azNew;
      }
      azResult[nRow] = sqlite3_mprintf("%s", sqlite3_column_text(pStmt, 0));
      if( 0==azResult[nRow] ){
        rc = shellNomemError();
        break;
      }
      nRow++;
    }
    if( sqlite3_finalize(pStmt)!=SQLITE_OK ){
      rc = shellDatabaseError(p->db);
    }

    /* Pretty-print the contents of array azResult[] to the output */
    if( rc==0 && nRow>0 ){
      int len, maxlen = 0;
      int i, j;
      int nPrintCol, nPrintRow;
      for(i=0; i<nRow; i++){
        len = strlen30(azResult[i]);
        if( len>maxlen ) maxlen = len;
      }
      nPrintCol = 80/(maxlen+2);
      if( nPrintCol<1 ) nPrintCol = 1;
      nPrintRow = (nRow + nPrintCol - 1)/nPrintCol;
      for(i=0; i<nPrintRow; i++){
        for(j=i; j<nRow; j+=nPrintRow){
          char *zSp = j<nPrintRow ? "" : "  ";
          utf8_printf(p->out, "%s%-*s", zSp, maxlen,
                      azResult[j] ? azResult[j]:"");
        }
        raw_printf(p->out, "\n");
      }
    }

    for(ii=0; ii<nRow; ii++) sqlite3_free(azResult[ii]);
    sqlite3_free(azResult);
  }else

  /* Begin redirecting output to the file "testcase-out.txt" */
  if( c=='t' && strcmp(azArg[0],"testcase")==0 ){
    output_reset(p);
    p->out = output_file_open("testcase-out.txt");
    if( p->out==0 ){
      raw_printf(stderr, "Error: cannot open 'testcase-out.txt'\n");
    }
    if( nArg>=2 ){
      sqlite3_snprintf(sizeof(p->zTestcase), p->zTestcase, "%s", azArg[1]);
    }else{
      sqlite3_snprintf(sizeof(p->zTestcase), p->zTestcase, "?");
    }
  }else

#ifndef SQLITE_UNTESTABLE
  if( c=='t' && n>=8 && strncmp(azArg[0], "testctrl", n)==0 && nArg>=2 ){
    static const struct {
       const char *zCtrlName;   /* Name of a test-control option */
       int ctrlCode;            /* Integer code for that option */
    } aCtrl[] = {
      { "prng_save",             SQLITE_TESTCTRL_PRNG_SAVE              },
      { "prng_restore",          SQLITE_TESTCTRL_PRNG_RESTORE           },
      { "prng_reset",            SQLITE_TESTCTRL_PRNG_RESET             },
      { "bitvec_test",           SQLITE_TESTCTRL_BITVEC_TEST            },
      { "fault_install",         SQLITE_TESTCTRL_FAULT_INSTALL          },
      { "benign_malloc_hooks",   SQLITE_TESTCTRL_BENIGN_MALLOC_HOOKS    },
      { "pending_byte",          SQLITE_TESTCTRL_PENDING_BYTE           },
      { "assert",                SQLITE_TESTCTRL_ASSERT                 },
      { "always",                SQLITE_TESTCTRL_ALWAYS                 },
      { "reserve",               SQLITE_TESTCTRL_RESERVE                },
      { "optimizations",         SQLITE_TESTCTRL_OPTIMIZATIONS          },
      { "iskeyword",             SQLITE_TESTCTRL_ISKEYWORD              },
      { "scratchmalloc",         SQLITE_TESTCTRL_SCRATCHMALLOC          },
      { "byteorder",             SQLITE_TESTCTRL_BYTEORDER              },
      { "never_corrupt",         SQLITE_TESTCTRL_NEVER_CORRUPT          },
      { "imposter",              SQLITE_TESTCTRL_IMPOSTER               },
    };
    int testctrl = -1;
    int rc2 = 0;
    int i, n2;
    open_db(p, 0);

    /* convert testctrl text option to value. allow any unique prefix
    ** of the option name, or a numerical value. */
    n2 = strlen30(azArg[1]);
    for(i=0; i<ArraySize(aCtrl); i++){
      if( strncmp(azArg[1], aCtrl[i].zCtrlName, n2)==0 ){
        if( testctrl<0 ){
          testctrl = aCtrl[i].ctrlCode;
        }else{
          utf8_printf(stderr, "ambiguous option name: \"%s\"\n", azArg[1]);
          testctrl = -1;
          break;
        }
      }
    }
    if( testctrl<0 ) testctrl = (int)integerValue(azArg[1]);
    if( (testctrl<SQLITE_TESTCTRL_FIRST) || (testctrl>SQLITE_TESTCTRL_LAST) ){
      utf8_printf(stderr,"Error: invalid testctrl option: %s\n", azArg[1]);
    }else{
      switch(testctrl){

        /* sqlite3_test_control(int, db, int) */
        case SQLITE_TESTCTRL_OPTIMIZATIONS:
        case SQLITE_TESTCTRL_RESERVE:
          if( nArg==3 ){
            int opt = (int)strtol(azArg[2], 0, 0);
            rc2 = sqlite3_test_control(testctrl, p->db, opt);
            raw_printf(p->out, "%d (0x%08x)\n", rc2, rc2);
          } else {
            utf8_printf(stderr,"Error: testctrl %s takes a single int option\n",
                    azArg[1]);
          }
          break;

        /* sqlite3_test_control(int) */
        case SQLITE_TESTCTRL_PRNG_SAVE:
        case SQLITE_TESTCTRL_PRNG_RESTORE:
        case SQLITE_TESTCTRL_PRNG_RESET:
        case SQLITE_TESTCTRL_BYTEORDER:
          if( nArg==2 ){
            rc2 = sqlite3_test_control(testctrl);
            raw_printf(p->out, "%d (0x%08x)\n", rc2, rc2);
          } else {
            utf8_printf(stderr,"Error: testctrl %s takes no options\n",
                        azArg[1]);
          }
          break;

        /* sqlite3_test_control(int, uint) */
        case SQLITE_TESTCTRL_PENDING_BYTE:
          if( nArg==3 ){
            unsigned int opt = (unsigned int)integerValue(azArg[2]);
            rc2 = sqlite3_test_control(testctrl, opt);
            raw_printf(p->out, "%d (0x%08x)\n", rc2, rc2);
          } else {
            utf8_printf(stderr,"Error: testctrl %s takes a single unsigned"
                           " int option\n", azArg[1]);
          }
          break;

        /* sqlite3_test_control(int, int) */
        case SQLITE_TESTCTRL_ASSERT:
        case SQLITE_TESTCTRL_ALWAYS:
        case SQLITE_TESTCTRL_NEVER_CORRUPT:
          if( nArg==3 ){
            int opt = booleanValue(azArg[2]);
            rc2 = sqlite3_test_control(testctrl, opt);
            raw_printf(p->out, "%d (0x%08x)\n", rc2, rc2);
          } else {
            utf8_printf(stderr,"Error: testctrl %s takes a single int option\n",
                            azArg[1]);
          }
          break;

        /* sqlite3_test_control(int, char *) */
#ifdef SQLITE_N_KEYWORD
        case SQLITE_TESTCTRL_ISKEYWORD:
          if( nArg==3 ){
            const char *opt = azArg[2];
            rc2 = sqlite3_test_control(testctrl, opt);
            raw_printf(p->out, "%d (0x%08x)\n", rc2, rc2);
          } else {
            utf8_printf(stderr,
                        "Error: testctrl %s takes a single char * option\n",
                        azArg[1]);
          }
          break;
#endif

        case SQLITE_TESTCTRL_IMPOSTER:
          if( nArg==5 ){
            rc2 = sqlite3_test_control(testctrl, p->db,
                          azArg[2],
                          integerValue(azArg[3]),
                          integerValue(azArg[4]));
            raw_printf(p->out, "%d (0x%08x)\n", rc2, rc2);
          }else{
            raw_printf(stderr,"Usage: .testctrl imposter dbName onoff tnum\n");
          }
          break;

        case SQLITE_TESTCTRL_BITVEC_TEST:
        case SQLITE_TESTCTRL_FAULT_INSTALL:
        case SQLITE_TESTCTRL_BENIGN_MALLOC_HOOKS:
        case SQLITE_TESTCTRL_SCRATCHMALLOC:
        default:
          utf8_printf(stderr,
                      "Error: CLI support for testctrl %s not implemented\n",
                      azArg[1]);
          break;
      }
    }
  }else
#endif /* !defined(SQLITE_UNTESTABLE) */

  if( c=='t' && n>4 && strncmp(azArg[0], "timeout", n)==0 ){
    open_db(p, 0);
    sqlite3_busy_timeout(p->db, nArg>=2 ? (int)integerValue(azArg[1]) : 0);
  }else

  if( c=='t' && n>=5 && strncmp(azArg[0], "timer", n)==0 ){
    if( nArg==2 ){
      enableTimer = booleanValue(azArg[1]);
      if( enableTimer && !HAS_TIMER ){
        raw_printf(stderr, "Error: timer not available on this system.\n");
        enableTimer = 0;
      }
    }else{
      raw_printf(stderr, "Usage: .timer on|off\n");
      rc = 1;
    }
  }else

  if( c=='t' && strncmp(azArg[0], "trace", n)==0 ){
    open_db(p, 0);
    if( nArg!=2 ){
      raw_printf(stderr, "Usage: .trace FILE|off\n");
      rc = 1;
      goto meta_command_exit;
    }
    output_file_close(p->traceOut);
    p->traceOut = output_file_open(azArg[1]);
#if !defined(SQLITE_OMIT_TRACE) && !defined(SQLITE_OMIT_FLOATING_POINT)
    if( p->traceOut==0 ){
      sqlite3_trace_v2(p->db, 0, 0, 0);
    }else{
      sqlite3_trace_v2(p->db, SQLITE_TRACE_STMT, sql_trace_callback,p->traceOut);
    }
#endif
  }else

#if SQLITE_USER_AUTHENTICATION
  if( c=='u' && strncmp(azArg[0], "user", n)==0 ){
    if( nArg<2 ){
      raw_printf(stderr, "Usage: .user SUBCOMMAND ...\n");
      rc = 1;
      goto meta_command_exit;
    }
    open_db(p, 0);
    if( strcmp(azArg[1],"login")==0 ){
      if( nArg!=4 ){
        raw_printf(stderr, "Usage: .user login USER PASSWORD\n");
        rc = 1;
        goto meta_command_exit;
      }
      rc = sqlite3_user_authenticate(p->db, azArg[2], azArg[3],
                                    (int)strlen(azArg[3]));
      if( rc ){
        utf8_printf(stderr, "Authentication failed for user %s\n", azArg[2]);
        rc = 1;
      }
    }else if( strcmp(azArg[1],"add")==0 ){
      if( nArg!=5 ){
        raw_printf(stderr, "Usage: .user add USER PASSWORD ISADMIN\n");
        rc = 1;
        goto meta_command_exit;
      }
      rc = sqlite3_user_add(p->db, azArg[2],
                            azArg[3], (int)strlen(azArg[3]),
                            booleanValue(azArg[4]));
      if( rc ){
        raw_printf(stderr, "User-Add failed: %d\n", rc);
        rc = 1;
      }
    }else if( strcmp(azArg[1],"edit")==0 ){
      if( nArg!=5 ){
        raw_printf(stderr, "Usage: .user edit USER PASSWORD ISADMIN\n");
        rc = 1;
        goto meta_command_exit;
      }
      rc = sqlite3_user_change(p->db, azArg[2],
                              azArg[3], (int)strlen(azArg[3]),
                              booleanValue(azArg[4]));
      if( rc ){
        raw_printf(stderr, "User-Edit failed: %d\n", rc);
        rc = 1;
      }
    }else if( strcmp(azArg[1],"delete")==0 ){
      if( nArg!=3 ){
        raw_printf(stderr, "Usage: .user delete USER\n");
        rc = 1;
        goto meta_command_exit;
      }
      rc = sqlite3_user_delete(p->db, azArg[2]);
      if( rc ){
        raw_printf(stderr, "User-Delete failed: %d\n", rc);
        rc = 1;
      }
    }else{
      raw_printf(stderr, "Usage: .user login|add|edit|delete ...\n");
      rc = 1;
      goto meta_command_exit;
    }
  }else
#endif /* SQLITE_USER_AUTHENTICATION */

  if( c=='v' && strncmp(azArg[0], "version", n)==0 ){
    utf8_printf(p->out, "SQLite %s %s\n" /*extra-version-info*/,
        sqlite3_libversion(), sqlite3_sourceid());
  }else

  if( c=='v' && strncmp(azArg[0], "vfsinfo", n)==0 ){
    const char *zDbName = nArg==2 ? azArg[1] : "main";
    sqlite3_vfs *pVfs = 0;
    if( p->db ){
      sqlite3_file_control(p->db, zDbName, SQLITE_FCNTL_VFS_POINTER, &pVfs);
      if( pVfs ){
        utf8_printf(p->out, "vfs.zName      = \"%s\"\n", pVfs->zName);
        raw_printf(p->out, "vfs.iVersion   = %d\n", pVfs->iVersion);
        raw_printf(p->out, "vfs.szOsFile   = %d\n", pVfs->szOsFile);
        raw_printf(p->out, "vfs.mxPathname = %d\n", pVfs->mxPathname);
      }
    }
  }else

  if( c=='v' && strncmp(azArg[0], "vfslist", n)==0 ){
    sqlite3_vfs *pVfs;
    sqlite3_vfs *pCurrent = 0;
    if( p->db ){
      sqlite3_file_control(p->db, "main", SQLITE_FCNTL_VFS_POINTER, &pCurrent);
    }
    for(pVfs=sqlite3_vfs_find(0); pVfs; pVfs=pVfs->pNext){
      utf8_printf(p->out, "vfs.zName      = \"%s\"%s\n", pVfs->zName,
           pVfs==pCurrent ? "  <--- CURRENT" : "");
      raw_printf(p->out, "vfs.iVersion   = %d\n", pVfs->iVersion);
      raw_printf(p->out, "vfs.szOsFile   = %d\n", pVfs->szOsFile);
      raw_printf(p->out, "vfs.mxPathname = %d\n", pVfs->mxPathname);
      if( pVfs->pNext ){
        raw_printf(p->out, "-----------------------------------\n");
      }
    }
  }else

  if( c=='v' && strncmp(azArg[0], "vfsname", n)==0 ){
    const char *zDbName = nArg==2 ? azArg[1] : "main";
    char *zVfsName = 0;
    if( p->db ){
      sqlite3_file_control(p->db, zDbName, SQLITE_FCNTL_VFSNAME, &zVfsName);
      if( zVfsName ){
        utf8_printf(p->out, "%s\n", zVfsName);
        sqlite3_free(zVfsName);
      }
    }
  }else

#if defined(SQLITE_DEBUG) && defined(SQLITE_ENABLE_WHERETRACE)
  if( c=='w' && strncmp(azArg[0], "wheretrace", n)==0 ){
    sqlite3WhereTrace = nArg>=2 ? booleanValue(azArg[1]) : 0xff;
  }else
#endif

  if( c=='w' && strncmp(azArg[0], "width", n)==0 ){
    int j;
    assert( nArg<=ArraySize(azArg) );
    for(j=1; j<nArg && j<ArraySize(p->colWidth); j++){
      p->colWidth[j-1] = (int)integerValue(azArg[j]);
    }
  }else

  {
    utf8_printf(stderr, "Error: unknown command or invalid arguments: "
      " \"%s\". Enter \".help\" for help\n", azArg[0]);
    rc = 1;
  }

meta_command_exit:
  if( p->outCount ){
    p->outCount--;
    if( p->outCount==0 ) output_reset(p);
  }
  return rc;
}

/*
** Return TRUE if a semicolon occurs anywhere in the first N characters
** of string z[].
*/
static int line_contains_semicolon(const char *z, int N){
  int i;
  for(i=0; i<N; i++){  if( z[i]==';' ) return 1; }
  return 0;
}

/*
** Test to see if a line consists entirely of whitespace.
*/
static int _all_whitespace(const char *z){
  for(; *z; z++){
    if( IsSpace(z[0]) ) continue;
    if( *z=='/' && z[1]=='*' ){
      z += 2;
      while( *z && (*z!='*' || z[1]!='/') ){ z++; }
      if( *z==0 ) return 0;
      z++;
      continue;
    }
    if( *z=='-' && z[1]=='-' ){
      z += 2;
      while( *z && *z!='\n' ){ z++; }
      if( *z==0 ) return 1;
      continue;
    }
    return 0;
  }
  return 1;
}

/*
** Return TRUE if the line typed in is an SQL command terminator other
** than a semi-colon.  The SQL Server style "go" command is understood
** as is the Oracle "/".
*/
static int line_is_command_terminator(const char *zLine){
  while( IsSpace(zLine[0]) ){ zLine++; };
  if( zLine[0]=='/' && _all_whitespace(&zLine[1]) ){
    return 1;  /* Oracle */
  }
  if( ToLower(zLine[0])=='g' && ToLower(zLine[1])=='o'
         && _all_whitespace(&zLine[2]) ){
    return 1;  /* SQL Server */
  }
  return 0;
}

/*
** Return true if zSql is a complete SQL statement.  Return false if it
** ends in the middle of a string literal or C-style comment.
*/
static int line_is_complete(char *zSql, int nSql){
  int rc;
  if( zSql==0 ) return 1;
  zSql[nSql] = ';';
  zSql[nSql+1] = 0;
  rc = sqlite3_complete(zSql);
  zSql[nSql] = 0;
  return rc;
}

/*
** Run a single line of SQL
*/
static int runOneSqlLine(ShellState *p, char *zSql, FILE *in, int startline){
  int rc;
  char *zErrMsg = 0;

  open_db(p, 0);
  if( ShellHasFlag(p,SHFLG_Backslash) ) resolve_backslashes(zSql);
  BEGIN_TIMER;
  rc = shell_exec(p->db, zSql, shell_callback, p, &zErrMsg);
  END_TIMER;
  if( rc || zErrMsg ){
    char zPrefix[100];
    if( in!=0 || !stdin_is_interactive ){
      sqlite3_snprintf(sizeof(zPrefix), zPrefix,
                       "Error: near line %d:", startline);
    }else{
      sqlite3_snprintf(sizeof(zPrefix), zPrefix, "Error:");
    }
    if( zErrMsg!=0 ){
      utf8_printf(stderr, "%s %s\n", zPrefix, zErrMsg);
      sqlite3_free(zErrMsg);
      zErrMsg = 0;
    }else{
      utf8_printf(stderr, "%s %s\n", zPrefix, sqlite3_errmsg(p->db));
    }
    return 1;
  }else if( ShellHasFlag(p, SHFLG_CountChanges) ){
    raw_printf(p->out, "changes: %3d   total_changes: %d\n",
            sqlite3_changes(p->db), sqlite3_total_changes(p->db));
  }
  return 0;
}


/*
** Read input from *in and process it.  If *in==0 then input
** is interactive - the user is typing it it.  Otherwise, input
** is coming from a file or device.  A prompt is issued and history
** is saved only if input is interactive.  An interrupt signal will
** cause this routine to exit immediately, unless input is interactive.
**
** Return the number of errors.
*/
static int process_input(ShellState *p, FILE *in){
  char *zLine = 0;          /* A single input line */
  char *zSql = 0;           /* Accumulated SQL text */
  int nLine;                /* Length of current line */
  int nSql = 0;             /* Bytes of zSql[] used */
  int nAlloc = 0;           /* Allocated zSql[] space */
  int nSqlPrior = 0;        /* Bytes of zSql[] used by prior line */
  int rc;                   /* Error code */
  int errCnt = 0;           /* Number of errors seen */
  int lineno = 0;           /* Current line number */
  int startline = 0;        /* Line number for start of current input */

  while( errCnt==0 || !bail_on_error || (in==0 && stdin_is_interactive) ){
    fflush(p->out);
    zLine = one_input_line(in, zLine, nSql>0);
    if( zLine==0 ){
      /* End of input */
      if( in==0 && stdin_is_interactive ) printf("\n");
      break;
    }
    if( seenInterrupt ){
      if( in!=0 ) break;
      seenInterrupt = 0;
    }
    lineno++;
    if( nSql==0 && _all_whitespace(zLine) ){
      if( ShellHasFlag(p, SHFLG_Echo) ) printf("%s\n", zLine);
      continue;
    }
    if( zLine && zLine[0]=='.' && nSql==0 ){
      if( ShellHasFlag(p, SHFLG_Echo) ) printf("%s\n", zLine);
      rc = do_meta_command(zLine, p);
      if( rc==2 ){ /* exit requested */
        break;
      }else if( rc ){
        errCnt++;
      }
      continue;
    }
    if( line_is_command_terminator(zLine) && line_is_complete(zSql, nSql) ){
      memcpy(zLine,";",2);
    }
    nLine = strlen30(zLine);
    if( nSql+nLine+2>=nAlloc ){
      nAlloc = nSql+nLine+100;
      zSql = realloc(zSql, nAlloc);
      if( zSql==0 ){
        raw_printf(stderr, "Error: out of memory\n");
        exit(1);
      }
    }
    nSqlPrior = nSql;
    if( nSql==0 ){
      int i;
      for(i=0; zLine[i] && IsSpace(zLine[i]); i++){}
      assert( nAlloc>0 && zSql!=0 );
      memcpy(zSql, zLine+i, nLine+1-i);
      startline = lineno;
      nSql = nLine-i;
    }else{
      zSql[nSql++] = '\n';
      memcpy(zSql+nSql, zLine, nLine+1);
      nSql += nLine;
    }
    if( nSql && line_contains_semicolon(&zSql[nSqlPrior], nSql-nSqlPrior)
                && sqlite3_complete(zSql) ){
      errCnt += runOneSqlLine(p, zSql, in, startline);
      nSql = 0;
      if( p->outCount ){
        output_reset(p);
        p->outCount = 0;
      }
    }else if( nSql && _all_whitespace(zSql) ){
      if( ShellHasFlag(p, SHFLG_Echo) ) printf("%s\n", zSql);
      nSql = 0;
    }
  }
  if( nSql && !_all_whitespace(zSql) ){
    runOneSqlLine(p, zSql, in, startline);
  }
  free(zSql);
  free(zLine);
  return errCnt>0;
}

/*
** Return a pathname which is the user's home directory.  A
** 0 return indicates an error of some kind.
*/
static char *find_home_dir(int clearFlag){
  static char *home_dir = NULL;
  if( clearFlag ){
    free(home_dir);
    home_dir = 0;
    return 0;
  }
  if( home_dir ) return home_dir;

#if !defined(_WIN32) && !defined(WIN32) && !defined(_WIN32_WCE) \
     && !defined(__RTP__) && !defined(_WRS_KERNEL)
  {
    struct passwd *pwent;
    uid_t uid = getuid();
    if( (pwent=getpwuid(uid)) != NULL) {
      home_dir = pwent->pw_dir;
    }
  }
#endif

#if defined(_WIN32_WCE)
  /* Windows CE (arm-wince-mingw32ce-gcc) does not provide getenv()
   */
  home_dir = "/";
#else

#if defined(_WIN32) || defined(WIN32)
  if (!home_dir) {
    home_dir = getenv("USERPROFILE");
  }
#endif

  if (!home_dir) {
    home_dir = getenv("HOME");
  }

#if defined(_WIN32) || defined(WIN32)
  if (!home_dir) {
    char *zDrive, *zPath;
    int n;
    zDrive = getenv("HOMEDRIVE");
    zPath = getenv("HOMEPATH");
    if( zDrive && zPath ){
      n = strlen30(zDrive) + strlen30(zPath) + 1;
      home_dir = malloc( n );
      if( home_dir==0 ) return 0;
      sqlite3_snprintf(n, home_dir, "%s%s", zDrive, zPath);
      return home_dir;
    }
    home_dir = "c:\\";
  }
#endif

#endif /* !_WIN32_WCE */

  if( home_dir ){
    int n = strlen30(home_dir) + 1;
    char *z = malloc( n );
    if( z ) memcpy(z, home_dir, n);
    home_dir = z;
  }

  return home_dir;
}

/*
** Read input from the file given by sqliterc_override.  Or if that
** parameter is NULL, take input from ~/.sqliterc
**
** Returns the number of errors.
*/
static void process_sqliterc(
  ShellState *p,                  /* Configuration data */
  const char *sqliterc_override   /* Name of config file. NULL to use default */
){
  char *home_dir = NULL;
  const char *sqliterc = sqliterc_override;
  char *zBuf = 0;
  FILE *in = NULL;

  if (sqliterc == NULL) {
    home_dir = find_home_dir(0);
    if( home_dir==0 ){
      raw_printf(stderr, "-- warning: cannot find home directory;"
                      " cannot read ~/.sqliterc\n");
      return;
    }
    sqlite3_initialize();
    zBuf = sqlite3_mprintf("%s/.sqliterc",home_dir);
    sqliterc = zBuf;
  }
  in = fopen(sqliterc,"rb");
  if( in ){
    if( stdin_is_interactive ){
      utf8_printf(stderr,"-- Loading resources from %s\n",sqliterc);
    }
    process_input(p,in);
    fclose(in);
  }
  sqlite3_free(zBuf);
}

/*
** Show available command line options
*/
static const char zOptions[] =
  "   -ascii               set output mode to 'ascii'\n"
  "   -bail                stop after hitting an error\n"
  "   -batch               force batch I/O\n"
  "   -column              set output mode to 'column'\n"
  "   -cmd COMMAND         run \"COMMAND\" before reading stdin\n"
  "   -csv                 set output mode to 'csv'\n"
  "   -echo                print commands before execution\n"
  "   -init FILENAME       read/process named file\n"
  "   -[no]header          turn headers on or off\n"
#if defined(SQLITE_ENABLE_MEMSYS3) || defined(SQLITE_ENABLE_MEMSYS5)
  "   -heap SIZE           Size of heap for memsys3 or memsys5\n"
#endif
  "   -help                show this message\n"
  "   -html                set output mode to HTML\n"
  "   -interactive         force interactive I/O\n"
  "   -line                set output mode to 'line'\n"
  "   -list                set output mode to 'list'\n"
  "   -lookaside SIZE N    use N entries of SZ bytes for lookaside memory\n"
  "   -mmap N              default mmap size set to N\n"
#ifdef SQLITE_ENABLE_MULTIPLEX
  "   -multiplex           enable the multiplexor VFS\n"
#endif
  "   -newline SEP         set output row separator. Default: '\\n'\n"
  "   -nullvalue TEXT      set text string for NULL values. Default ''\n"
  "   -pagecache SIZE N    use N slots of SZ bytes each for page cache memory\n"
  "   -scratch SIZE N      use N slots of SZ bytes each for scratch memory\n"
  "   -separator SEP       set output column separator. Default: '|'\n"
  "   -stats               print memory stats before each finalize\n"
  "   -version             show SQLite version\n"
  "   -vfs NAME            use NAME as the default VFS\n"
#ifdef SQLITE_ENABLE_VFSTRACE
  "   -vfstrace            enable tracing of all VFS calls\n"
#endif
;
static void usage(int showDetail){
  utf8_printf(stderr,
      "Usage: %s [OPTIONS] FILENAME [SQL]\n"
      "FILENAME is the name of an SQLite database. A new database is created\n"
      "if the file does not previously exist.\n", Argv0);
  if( showDetail ){
    utf8_printf(stderr, "OPTIONS include:\n%s", zOptions);
  }else{
    raw_printf(stderr, "Use the -help option for additional information\n");
  }
  exit(1);
}

/*
** Initialize the state information in data
*/
static void main_init(ShellState *data) {
  memset(data, 0, sizeof(*data));
  data->normalMode = data->cMode = data->mode = MODE_List;
  data->autoExplain = 1;
  memcpy(data->colSeparator,SEP_Column, 2);
  memcpy(data->rowSeparator,SEP_Row, 2);
  data->showHeader = 0;
  data->shellFlgs = SHFLG_Lookaside;
  sqlite3_config(SQLITE_CONFIG_URI, 1);
  sqlite3_config(SQLITE_CONFIG_LOG, shellLog, data);
  sqlite3_config(SQLITE_CONFIG_MULTITHREAD);
  sqlite3_snprintf(sizeof(mainPrompt), mainPrompt,"sqlite> ");
  sqlite3_snprintf(sizeof(continuePrompt), continuePrompt,"   ...> ");
}

/*
** Output text to the console in a font that attracts extra attention.
*/
#ifdef _WIN32
static void printBold(const char *zText){
  HANDLE out = GetStdHandle(STD_OUTPUT_HANDLE);
  CONSOLE_SCREEN_BUFFER_INFO defaultScreenInfo;
  GetConsoleScreenBufferInfo(out, &defaultScreenInfo);
  SetConsoleTextAttribute(out,
         FOREGROUND_RED|FOREGROUND_INTENSITY
  );
  printf("%s", zText);
  SetConsoleTextAttribute(out, defaultScreenInfo.wAttributes);
}
#else
static void printBold(const char *zText){
  printf("\033[1m%s\033[0m", zText);
}
#endif

/*
** Get the argument to an --option.  Throw an error and die if no argument
** is available.
*/
static char *cmdline_option_value(int argc, char **argv, int i){
  if( i==argc ){
    utf8_printf(stderr, "%s: Error: missing argument to %s\n",
            argv[0], argv[argc-1]);
    exit(1);
  }
  return argv[i];
}

#ifndef SQLITE_SHELL_IS_UTF8
#  if (defined(_WIN32) || defined(WIN32)) && defined(_MSC_VER)
#    define SQLITE_SHELL_IS_UTF8          (0)
#  else
#    define SQLITE_SHELL_IS_UTF8          (1)
#  endif
#endif

#if SQLITE_SHELL_IS_UTF8
int SQLITE_CDECL main(int argc, char **argv){
#else
int SQLITE_CDECL wmain(int argc, wchar_t **wargv){
  char **argv;
#endif
  char *zErrMsg = 0;
  ShellState data;
  const char *zInitFile = 0;
  int i;
  int rc = 0;
  int warnInmemoryDb = 0;
  int readStdin = 1;
  int nCmd = 0;
  char **azCmd = 0;

  setBinaryMode(stdin, 0);
  setvbuf(stderr, 0, _IONBF, 0); /* Make sure stderr is unbuffered */
  stdin_is_interactive = isatty(0);
  stdout_is_console = isatty(1);

#if USE_SYSTEM_SQLITE+0!=1
  if( strcmp(sqlite3_sourceid(),SQLITE_SOURCE_ID)!=0 ){
    utf8_printf(stderr, "SQLite header and source version mismatch\n%s\n%s\n",
            sqlite3_sourceid(), SQLITE_SOURCE_ID);
    exit(1);
  }
#endif
  main_init(&data);
#if !SQLITE_SHELL_IS_UTF8
  sqlite3_initialize();
  argv = sqlite3_malloc64(sizeof(argv[0])*argc);
  if( argv==0 ){
    raw_printf(stderr, "out of memory\n");
    exit(1);
  }
  for(i=0; i<argc; i++){
    argv[i] = sqlite3_win32_unicode_to_utf8(wargv[i]);
    if( argv[i]==0 ){
      raw_printf(stderr, "out of memory\n");
      exit(1);
    }
  }
#endif
  assert( argc>=1 && argv && argv[0] );
  Argv0 = argv[0];

  /* Make sure we have a valid signal handler early, before anything
  ** else is done.
  */
#ifdef SIGINT
  signal(SIGINT, interrupt_handler);
#endif

#ifdef SQLITE_SHELL_DBNAME_PROC
  {
    /* If the SQLITE_SHELL_DBNAME_PROC macro is defined, then it is the name
    ** of a C-function that will provide the name of the database file.  Use
    ** this compile-time option to embed this shell program in larger
    ** applications. */
    extern void SQLITE_SHELL_DBNAME_PROC(const char**);
    SQLITE_SHELL_DBNAME_PROC(&data.zDbFilename);
    warnInmemoryDb = 0;
  }
#endif

  /* Do an initial pass through the command-line argument to locate
  ** the name of the database file, the name of the initialization file,
  ** the size of the alternative malloc heap,
  ** and the first command to execute.
  */
  for(i=1; i<argc; i++){
    char *z;
    z = argv[i];
    if( z[0]!='-' ){
      if( data.zDbFilename==0 ){
        data.zDbFilename = z;
      }else{
        /* Excesss arguments are interpreted as SQL (or dot-commands) and
        ** mean that nothing is read from stdin */
        readStdin = 0;
        nCmd++;
        azCmd = realloc(azCmd, sizeof(azCmd[0])*nCmd);
        if( azCmd==0 ){
          raw_printf(stderr, "out of memory\n");
          exit(1);
        }
        azCmd[nCmd-1] = z;
      }
    }
    if( z[1]=='-' ) z++;
    if( strcmp(z,"-separator")==0
     || strcmp(z,"-nullvalue")==0
     || strcmp(z,"-newline")==0
     || strcmp(z,"-cmd")==0
    ){
      (void)cmdline_option_value(argc, argv, ++i);
    }else if( strcmp(z,"-init")==0 ){
      zInitFile = cmdline_option_value(argc, argv, ++i);
    }else if( strcmp(z,"-batch")==0 ){
      /* Need to check for batch mode here to so we can avoid printing
      ** informational messages (like from process_sqliterc) before
      ** we do the actual processing of arguments later in a second pass.
      */
      stdin_is_interactive = 0;
    }else if( strcmp(z,"-heap")==0 ){
#if defined(SQLITE_ENABLE_MEMSYS3) || defined(SQLITE_ENABLE_MEMSYS5)
      const char *zSize;
      sqlite3_int64 szHeap;

      zSize = cmdline_option_value(argc, argv, ++i);
      szHeap = integerValue(zSize);
      if( szHeap>0x7fff0000 ) szHeap = 0x7fff0000;
      sqlite3_config(SQLITE_CONFIG_HEAP, malloc((int)szHeap), (int)szHeap, 64);
#else
      (void)cmdline_option_value(argc, argv, ++i);
#endif
    }else if( strcmp(z,"-scratch")==0 ){
      int n, sz;
      sz = (int)integerValue(cmdline_option_value(argc,argv,++i));
      if( sz>400000 ) sz = 400000;
      if( sz<2500 ) sz = 2500;
      n = (int)integerValue(cmdline_option_value(argc,argv,++i));
      if( n>10 ) n = 10;
      if( n<1 ) n = 1;
      sqlite3_config(SQLITE_CONFIG_SCRATCH, malloc(n*sz+1), sz, n);
      data.shellFlgs |= SHFLG_Scratch;
    }else if( strcmp(z,"-pagecache")==0 ){
      int n, sz;
      sz = (int)integerValue(cmdline_option_value(argc,argv,++i));
      if( sz>70000 ) sz = 70000;
      if( sz<0 ) sz = 0;
      n = (int)integerValue(cmdline_option_value(argc,argv,++i));
      sqlite3_config(SQLITE_CONFIG_PAGECACHE,
                    (n>0 && sz>0) ? malloc(n*sz) : 0, sz, n);
      data.shellFlgs |= SHFLG_Pagecache;
    }else if( strcmp(z,"-lookaside")==0 ){
      int n, sz;
      sz = (int)integerValue(cmdline_option_value(argc,argv,++i));
      if( sz<0 ) sz = 0;
      n = (int)integerValue(cmdline_option_value(argc,argv,++i));
      if( n<0 ) n = 0;
      sqlite3_config(SQLITE_CONFIG_LOOKASIDE, sz, n);
      if( sz*n==0 ) data.shellFlgs &= ~SHFLG_Lookaside;
#ifdef SQLITE_ENABLE_VFSTRACE
    }else if( strcmp(z,"-vfstrace")==0 ){
      extern int vfstrace_register(
         const char *zTraceName,
         const char *zOldVfsName,
         int (*xOut)(const char*,void*),
         void *pOutArg,
         int makeDefault
      );
      vfstrace_register("trace",0,(int(*)(const char*,void*))fputs,stderr,1);
#endif
#ifdef SQLITE_ENABLE_MULTIPLEX
    }else if( strcmp(z,"-multiplex")==0 ){
      extern int sqlite3_multiple_initialize(const char*,int);
      sqlite3_multiplex_initialize(0, 1);
#endif
    }else if( strcmp(z,"-mmap")==0 ){
      sqlite3_int64 sz = integerValue(cmdline_option_value(argc,argv,++i));
      sqlite3_config(SQLITE_CONFIG_MMAP_SIZE, sz, sz);
    }else if( strcmp(z,"-vfs")==0 ){
      sqlite3_vfs *pVfs = sqlite3_vfs_find(cmdline_option_value(argc,argv,++i));
      if( pVfs ){
        sqlite3_vfs_register(pVfs, 1);
      }else{
        utf8_printf(stderr, "no such VFS: \"%s\"\n", argv[i]);
        exit(1);
      }
    }
  }
  if( data.zDbFilename==0 ){
#ifndef SQLITE_OMIT_MEMORYDB
    data.zDbFilename = ":memory:";
    warnInmemoryDb = argc==1;
#else
    utf8_printf(stderr,"%s: Error: no database filename specified\n", Argv0);
    return 1;
#endif
  }
  data.out = stdout;

  /* Go ahead and open the database file if it already exists.  If the
  ** file does not exist, delay opening it.  This prevents empty database
  ** files from being created if a user mistypes the database name argument
  ** to the sqlite command-line tool.
  */
  if( access(data.zDbFilename, 0)==0 ){
    open_db(&data, 0);
  }

  /* Process the initialization file if there is one.  If no -init option
  ** is given on the command line, look for a file named ~/.sqliterc and
  ** try to process it.
  */
  process_sqliterc(&data,zInitFile);

  /* Make a second pass through the command-line argument and set
  ** options.  This second pass is delayed until after the initialization
  ** file is processed so that the command-line arguments will override
  ** settings in the initialization file.
  */
  for(i=1; i<argc; i++){
    char *z = argv[i];
    if( z[0]!='-' ) continue;
    if( z[1]=='-' ){ z++; }
    if( strcmp(z,"-init")==0 ){
      i++;
    }else if( strcmp(z,"-html")==0 ){
      data.mode = MODE_Html;
    }else if( strcmp(z,"-list")==0 ){
      data.mode = MODE_List;
    }else if( strcmp(z,"-line")==0 ){
      data.mode = MODE_Line;
    }else if( strcmp(z,"-column")==0 ){
      data.mode = MODE_Column;
    }else if( strcmp(z,"-csv")==0 ){
      data.mode = MODE_Csv;
      memcpy(data.colSeparator,",",2);
    }else if( strcmp(z,"-ascii")==0 ){
      data.mode = MODE_Ascii;
      sqlite3_snprintf(sizeof(data.colSeparator), data.colSeparator,
                       SEP_Unit);
      sqlite3_snprintf(sizeof(data.rowSeparator), data.rowSeparator,
                       SEP_Record);
    }else if( strcmp(z,"-separator")==0 ){
      sqlite3_snprintf(sizeof(data.colSeparator), data.colSeparator,
                       "%s",cmdline_option_value(argc,argv,++i));
    }else if( strcmp(z,"-newline")==0 ){
      sqlite3_snprintf(sizeof(data.rowSeparator), data.rowSeparator,
                       "%s",cmdline_option_value(argc,argv,++i));
    }else if( strcmp(z,"-nullvalue")==0 ){
      sqlite3_snprintf(sizeof(data.nullValue), data.nullValue,
                       "%s",cmdline_option_value(argc,argv,++i));
    }else if( strcmp(z,"-header")==0 ){
      data.showHeader = 1;
    }else if( strcmp(z,"-noheader")==0 ){
      data.showHeader = 0;
    }else if( strcmp(z,"-echo")==0 ){
      ShellSetFlag(&data, SHFLG_Echo);
    }else if( strcmp(z,"-eqp")==0 ){
      data.autoEQP = 1;
    }else if( strcmp(z,"-eqpfull")==0 ){
      data.autoEQP = 2;
    }else if( strcmp(z,"-stats")==0 ){
      data.statsOn = 1;
    }else if( strcmp(z,"-scanstats")==0 ){
      data.scanstatsOn = 1;
    }else if( strcmp(z,"-backslash")==0 ){
      /* Undocumented command-line option: -backslash
      ** Causes C-style backslash escapes to be evaluated in SQL statements
      ** prior to sending the SQL into SQLite.  Useful for injecting
      ** crazy bytes in the middle of SQL statements for testing and debugging.
      */
      ShellSetFlag(&data, SHFLG_Backslash);
    }else if( strcmp(z,"-bail")==0 ){
      bail_on_error = 1;
    }else if( strcmp(z,"-version")==0 ){
      printf("%s %s\n", sqlite3_libversion(), sqlite3_sourceid());
      return 0;
    }else if( strcmp(z,"-interactive")==0 ){
      stdin_is_interactive = 1;
    }else if( strcmp(z,"-batch")==0 ){
      stdin_is_interactive = 0;
    }else if( strcmp(z,"-heap")==0 ){
      i++;
    }else if( strcmp(z,"-scratch")==0 ){
      i+=2;
    }else if( strcmp(z,"-pagecache")==0 ){
      i+=2;
    }else if( strcmp(z,"-lookaside")==0 ){
      i+=2;
    }else if( strcmp(z,"-mmap")==0 ){
      i++;
    }else if( strcmp(z,"-vfs")==0 ){
      i++;
#ifdef SQLITE_ENABLE_VFSTRACE
    }else if( strcmp(z,"-vfstrace")==0 ){
      i++;
#endif
#ifdef SQLITE_ENABLE_MULTIPLEX
    }else if( strcmp(z,"-multiplex")==0 ){
      i++;
#endif
    }else if( strcmp(z,"-help")==0 ){
      usage(1);
    }else if( strcmp(z,"-cmd")==0 ){
      /* Run commands that follow -cmd first and separately from commands
      ** that simply appear on the command-line.  This seems goofy.  It would
      ** be better if all commands ran in the order that they appear.  But
      ** we retain the goofy behavior for historical compatibility. */
      if( i==argc-1 ) break;
      z = cmdline_option_value(argc,argv,++i);
      if( z[0]=='.' ){
        rc = do_meta_command(z, &data);
        if( rc && bail_on_error ) return rc==2 ? 0 : rc;
      }else{
        open_db(&data, 0);
        rc = shell_exec(data.db, z, shell_callback, &data, &zErrMsg);
        if( zErrMsg!=0 ){
          utf8_printf(stderr,"Error: %s\n", zErrMsg);
          if( bail_on_error ) return rc!=0 ? rc : 1;
        }else if( rc!=0 ){
          utf8_printf(stderr,"Error: unable to process SQL \"%s\"\n", z);
          if( bail_on_error ) return rc;
        }
      }
    }else{
      utf8_printf(stderr,"%s: Error: unknown option: %s\n", Argv0, z);
      raw_printf(stderr,"Use -help for a list of options.\n");
      return 1;
    }
    data.cMode = data.mode;
  }

  if( !readStdin ){
    /* Run all arguments that do not begin with '-' as if they were separate
    ** command-line inputs, except for the argToSkip argument which contains
    ** the database filename.
    */
    for(i=0; i<nCmd; i++){
      if( azCmd[i][0]=='.' ){
        rc = do_meta_command(azCmd[i], &data);
        if( rc ) return rc==2 ? 0 : rc;
      }else{
        open_db(&data, 0);
        rc = shell_exec(data.db, azCmd[i], shell_callback, &data, &zErrMsg);
        if( zErrMsg!=0 ){
          utf8_printf(stderr,"Error: %s\n", zErrMsg);
          return rc!=0 ? rc : 1;
        }else if( rc!=0 ){
          utf8_printf(stderr,"Error: unable to process SQL: %s\n", azCmd[i]);
          return rc;
        }
      }
    }
    free(azCmd);
  }else{
    /* Run commands received from standard input
    */
    if( stdin_is_interactive ){
      char *zHome;
      char *zHistory = 0;
      int nHistory;
      printf(
        "SQLite version %s %.19s\n" /*extra-version-info*/
        "Enter \".help\" for usage hints.\n",
        sqlite3_libversion(), sqlite3_sourceid()
      );
      if( warnInmemoryDb ){
        printf("Connected to a ");
        printBold("transient in-memory database");
        printf(".\nUse \".open FILENAME\" to reopen on a "
               "persistent database.\n");
      }
      zHome = find_home_dir(0);
      if( zHome ){
        nHistory = strlen30(zHome) + 20;
        if( (zHistory = malloc(nHistory))!=0 ){
          sqlite3_snprintf(nHistory, zHistory,"%s/.sqlite_history", zHome);
        }
      }
      if( zHistory ){ shell_read_history(zHistory); }
      rc = process_input(&data, 0);
      if( zHistory ){
        shell_stifle_history(100);
        shell_write_history(zHistory);
        free(zHistory);
      }
    }else{
      rc = process_input(&data, stdin);
    }
  }
  set_table_name(&data, 0);
  if( data.db ){
    session_close_all(&data);
    sqlite3_close(data.db);
  }
  sqlite3_free(data.zFreeOnClose);
  find_home_dir(1);
#if !SQLITE_SHELL_IS_UTF8
  for(i=0; i<argc; i++) sqlite3_free(argv[i]);
  sqlite3_free(argv);
#endif
  return rc;
}
