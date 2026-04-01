MODULE_NAME := snd-soc-fpgafft-codec
OVERLAY_NAME := fpga-i2s-rx-32x2-slave

obj-m += $(MODULE_NAME).o

KERNEL_RELEASE ?= $(shell uname -r)
KERNEL_BUILD_DIR ?= /lib/modules/$(KERNEL_RELEASE)/build
DTC ?= dtc

.PHONY: all module overlay clean help

all: module overlay

module:
	$(MAKE) -C $(KERNEL_BUILD_DIR) M=$(CURDIR) modules

overlay: $(OVERLAY_NAME).dtbo

$(OVERLAY_NAME).dtbo: $(OVERLAY_NAME)-overlay.dts
	$(DTC) -@ -I dts -O dtb -o $@ $<

clean:
	@if [ -d "$(KERNEL_BUILD_DIR)" ]; then \
		$(MAKE) -C $(KERNEL_BUILD_DIR) M=$(CURDIR) clean; \
	fi
	rm -f $(OVERLAY_NAME).dtbo

help:
	@echo "Targets:"
	@echo "  make all     - build the codec module and overlay"
	@echo "  make module  - build $(MODULE_NAME).ko"
	@echo "  make overlay - build $(OVERLAY_NAME).dtbo"
	@echo "  make clean   - clean build outputs"
	@echo
	@echo "Variables:"
	@echo "  KERNEL_RELEASE   Kernel release used for /lib/modules/<release>/build"
	@echo "  KERNEL_BUILD_DIR Kernel build directory (default: /lib/modules/\$$KERNEL_RELEASE/build)"
	@echo "  DTC              Device Tree compiler executable"
