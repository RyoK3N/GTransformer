#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <signal.h>
#include <string.h>

#define CPU_LIMIT 90  //  Kill if CPU > 90%
#define MEM_LIMIT 70  //  Kill if memory > 70%

// Function to get CPU usage of a running process
int get_cpu_usage(pid_t pid) {
    char stat_file[256];
    sprintf(stat_file, "/proc/%d/stat", pid);

    FILE *fp = fopen(stat_file, "r");
    if (!fp) return -1;

    long utime, stime;
    for (int i = 0; i < 13; ++i) fscanf(fp, "%*s"); // Skip first 13 fields
    fscanf(fp, "%ld %ld", &utime, &stime);

    fclose(fp);
    return (utime + stime) / sysconf(_SC_CLK_TCK); 
}

// Function to get memory usage of a running process
int get_memory_usage(pid_t pid) {
    char statm_file[256];
    sprintf(statm_file, "/proc/%d/statm", pid);

    FILE *fp = fopen(statm_file, "r");
    if (!fp) return -1;

    long mem_pages;
    fscanf(fp, "%ld", &mem_pages);
    fclose(fp);

    long page_size = sysconf(_SC_PAGESIZE);
    return (mem_pages * page_size) / (1024 * 1024); //in Mb
}

int main() {
    // Read PID of the Python program which is running the model
    pid_t pid;
    FILE *pid_file = fopen("python_program.pid", "r");
    if (!pid_file) {
        fprintf(stderr, "PID file not found\n");
        return 1;
    }
    fscanf(pid_file, "%d", &pid);
    fclose(pid_file);

    while (1) {
        int cpu = get_cpu_usage(pid);
        int mem = get_memory_usage(pid);

        if (cpu < 0 || mem < 0) {
            fprintf(stderr, "Error reading process info\n");
            break;
        }

        printf("CPU: %d%%, MEM: %dMB\n", cpu, mem);

        if (cpu > CPU_LIMIT || mem > MEM_LIMIT) {
            printf("Resource limit exceeded. Killing process %d\n", pid);
            kill(pid, SIGKILL);
            break;
        }

        sleep(2); // Check every 2 seconds
    }

    return 0;
}