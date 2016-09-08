/*
Copyright (c) 2006, Michael Kazhdan and Matthew Bolitho
All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

Redistributions of source code must retain the above copyright notice, this list of
conditions and the following disclaimer. Redistributions in binary form must reproduce
the above copyright notice, this list of conditions and the following disclaimer
in the documentation and/or other materials provided with the distribution. 

Neither the name of the Johns Hopkins University nor the names of its contributors
may be used to endorse or promote products derived from this software without specific
prior written permission. 

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY
EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO THE IMPLIED WARRANTIES 
OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT
SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
TO, PROCUREMENT OF SUBSTITUTE  GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH
DAMAGE.
*/

#ifndef MEMORY_USAGE_INCLUDED
#define MEMORY_USAGE_INCLUDED

#ifdef WIN32

#include <Windows.h>
class MemoryInfo
{
public:
	size_t TotalPhysicalMemory;
	size_t FreePhysicalMemory;
	size_t TotalSwapSpace;
	size_t FreeSwapSpace;
	size_t TotalVirtualAddressSpace;
	size_t FreeVirtualAddressSpace;
	size_t PageSize;

	void set(void){
		MEMORYSTATUSEX Mem;
		SYSTEM_INFO Info;
		ZeroMemory( &Mem, sizeof(Mem));
		ZeroMemory( &Info, sizeof(Info)); 
		Mem.dwLength = sizeof(Mem);
		::GlobalMemoryStatusEx( &Mem );
		::GetSystemInfo( &Info );

		TotalPhysicalMemory = (size_t)Mem.ullTotalPhys;
		FreePhysicalMemory = (size_t)Mem.ullAvailPhys;
		TotalSwapSpace = (size_t)Mem.ullTotalPageFile;
		FreeSwapSpace = (size_t)Mem.ullAvailPageFile;
		TotalVirtualAddressSpace = (size_t)Mem.ullTotalVirtual;
		FreeVirtualAddressSpace = (size_t)Mem.ullAvailVirtual;
		PageSize = (size_t)Info.dwPageSize;
	}
	size_t usage(void) const {return TotalVirtualAddressSpace-FreeVirtualAddressSpace;}

	static size_t Usage(void){
		MEMORY_BASIC_INFORMATION mbi; 
		size_t      dwMemUsed = 0; 
		PVOID      pvAddress = 0; 


		memset(&mbi, 0, sizeof(MEMORY_BASIC_INFORMATION)); 
		while(VirtualQuery(pvAddress, &mbi, sizeof(MEMORY_BASIC_INFORMATION)) == sizeof(MEMORY_BASIC_INFORMATION)){ 
			if(mbi.State == MEM_COMMIT && mbi.Type == MEM_PRIVATE){dwMemUsed += mbi.RegionSize;}
			pvAddress = ((BYTE*)mbi.BaseAddress) + mbi.RegionSize; 
		} 
		return dwMemUsed; 
	} 
};

#else // !WIN32

#ifndef __APPLE__               // Linux variants

#include <sys/time.h>
#include <sys/resource.h>

class MemoryInfo
{
 public:
  static size_t Usage(void)
  {
		FILE* f = fopen("/proc/self/stat","rb");
		
		int d;
		long ld;
		unsigned long lu;
		unsigned long long llu;
		char s[1024];
		char c;
		
		int pid;
		unsigned long vm;

		int n = fscanf(f, "%d %s %c %d %d %d %d %d %lu %lu %lu %lu %lu %lu %lu %ld %ld %ld %ld %d %ld %llu %lu %ld %lu %lu %lu %lu %lu %lu %lu %lu %lu %lu %lu %lu %lu %d %d %lu %lu"
			,&pid ,s ,&c ,&d ,&d ,&d ,&d ,&d ,&lu ,&lu ,&lu ,&lu ,&lu ,&lu ,&lu ,&ld ,&ld ,&ld ,&ld ,&d ,&ld ,&llu ,&vm ,&ld ,&lu ,&lu ,&lu ,&lu ,&lu ,&lu ,&lu ,&lu ,&lu ,&lu ,&lu ,&lu ,&lu ,&d ,&d ,&lu ,&lu );

		fclose(f);
/*
pid %d 
comm %s 
state %c 
ppid %d 
pgrp %d 
session %d 
tty_nr %d
tpgid %d 
flags %lu 
minflt %lu 
cminflt %lu 
majflt %lu 
cmajflt %lu 
utime %lu 
stime %lu 
cutime %ld 
cstime %ld 
priority %ld 
nice %ld 
0 %ld 
itrealvalue %ld 
starttime %lu 
vsize %lu 
rss %ld 
rlim %lu 
startcode %lu 
endcode %lu 
startstack %lu 
kstkesp %lu 
kstkeip %lu 
signal %lu 
blocked %lu 
sigignore %lu 
sigcatch %lu 
wchan %lu 
nswap %lu 
cnswap %lu 
exit_signal %d 
processor %d 
rt_priority %lu (since kernel 2.5.19) 
policy %lu (since kernel 2.5.19) 
*/
		return vm;
	}

};
#else // __APPLE__: has no "/proc" pseudo-file system

// Thanks to David O'Gwynn for providing this fix.
// This comes from a post by Michael Knight:
//
// http://miknight.blogspot.com/2005/11/resident-set-size-in-mac-os-x.html

#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/sysctl.h>
#include <mach/task.h>
#include <mach/mach_init.h>

void getres(task_t task, unsigned long *rss, unsigned long *vs)
{
    struct task_basic_info t_info;
    mach_msg_type_number_t t_info_count = TASK_BASIC_INFO_COUNT;

    task_info(task, TASK_BASIC_INFO, (task_info_t)&t_info, &t_info_count);
    *rss = t_info.resident_size;
    *vs = t_info.virtual_size;
}

class MemoryInfo
{
 public:
  static size_t Usage(void)
  {
    unsigned long rss, vs, psize;
    task_t task = MACH_PORT_NULL;

    if (task_for_pid(current_task(), getpid(), &task) != KERN_SUCCESS)
        abort();
    getres(task, &rss, &vs);
    return rss;
  }

};

#endif // !__APPLE__  

#endif // WIN32

#endif // MEMORY_USAGE_INCLUDE
