/* nvidia-set-fanspeed.c */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <nvml.h>

int main(int argc, char **argv)
{
	nvmlReturn_t result;
	
	if (argc != 3) {
		fprintf(stderr, "Usage: %s <GPU index> <fan speed>\n", argv[0]);
		return 1;
	}

	nvmlInit();

	uint32_t device_count;
	result = nvmlDeviceGetCount(&device_count);
	if (NVML_SUCCESS != result) {
		fprintf(stderr, "Failed to query device count: %s\n", nvmlErrorString(result));
		return 1;
	}

	uint32_t device_index = atoi(argv[1]);
	if (device_index >= device_count) {
		fprintf(stderr, "Invalid device index %d\n", device_index);
		return 1;
	}

	nvmlDevice_t device;
	result = nvmlDeviceGetHandleByIndex(device_index, &device);
	if (NVML_SUCCESS != result) {
		fprintf(stderr, "Failed to get device handle for device %d: %s\n", device_index, nvmlErrorString(result));
		return 1;
	}

	// -1 is auto, 0 is 0%, 100 is 100%
	int fan_speed = atoi(argv[2]);
	if (fan_speed > 100) {
		fprintf(stderr, "Invalid fan speed %d\n", fan_speed);
		return 1;
	}
	if(fan_speed < -1) {
		fprintf(stderr, "Invalid fan speed %d\n", fan_speed);
		return 1;
	}

	// hardcode fan index to 0, because there's only one fan control on most GPUs
	nvmlFanControlPolicy_t policy = NVML_FAN_POLICY_MANUAL;
	if(fan_speed == -1) {
		policy = NVML_FAN_POLICY_TEMPERATURE_CONTINOUS_SW;
	}
	result = nvmlDeviceSetFanControlPolicy(device, 0, policy);
	if(NVML_SUCCESS != result) {
		fprintf(stderr, "Failed to set fan speed mode: %s\n", nvmlErrorString(result));
		return 1;
	}
	if(fan_speed != -1) {
		result = nvmlDeviceSetFanSpeed_v2(device, 0, fan_speed);
		if(NVML_SUCCESS != result) {
			fprintf(stderr, "Failed to set fan speed: %s\n", nvmlErrorString(result));
			return 1;
		}
	}

	return 0;
}
