#!/bin/bash
python analyzer_from_fpga_fft.py --capture-realign-tagged --record-button-line 17 --similarity-led-line 27 --similarity-led-hold-seconds 5 --min-db 90 --no-bfpexp
