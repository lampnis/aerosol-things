# DISCLAIMER:
!!!
USE WITH CAUTION,
CREATED WITH THE HELP OF AI,
BASED ON HUMAN-WRITTEN NOTES
!!!

# Project Worklog
This document tracks the progress, notes, and key findings of the aerosol project. It is generated from the `notes.txt` file.

## Undated Initial Notes

**Summary:** The initial notes cover leak testing, tuning of phase and amplitude parameters for the legacy .vi system at various pressures, and conceptualizing a script for automatic measurements. Key questions were raised about the necessity of multiple gas exchanges in a non-leaking cell and the need for averaging multiple spectra.

<details>
<summary>Original Notes</summary>

```
b# LAB NOTES

## Testing the leaks

1. w/o changes: for now the best (20 mbar/min)

2. w/ plug: worse (75 mbar/min)

3. w/ metal plug (2 screw plugs): instantly 1 atm after turning off the pump

# Tuning phase/amplitude parameters in legacy .vi @ ~ 19 [degC]

1. ~250 [mbar]
	phase 3rd 0x5550; phase 2nd 0x48C1; amplitude 0x47;

2. ~500 [mbar]
	phase 3rd 0x5200; phase 2nd 0x5041; amplitude 0x49;

3. ~750 [mbar]
	phase 3rd 0x5000; phase 2nd 0x5000; amplitude 0x5A;

4. 1 atm.
	phase 3r 0x5990; phase 2nd 0x48C1; amplitude 0x50;

TODO: double test these params for both peaks!!
250 [mbar]:
	OK
500 [mbar]:
	OK (peak intensity around 0.13 arb.u.)
750 [mbar]"
	OK (~0.13 arb.u.)

Concept of automatic measurements:

1) Measurement time = (DFB max tune - DFB offset 1) * Meas Time [s]

2) Loop for repeating measurements:
	def auto_measurements(n_measurements):
		for i in range(n_measurements):
			change_file_name(f"measurement_{i}")
			measure = True
			if measure == False:
				GasX or clear_chart
				continue

Questions:
1) If cell is not leaking, what variable would we need to loop over? Current setup is good for leaky cell,
because it does GasX multiple times per one spectrum recording. But if cell is not leaking much, there is no
need for multiple GasX per one spectrum.

2) Is there a need for multiple spectra for averaging? As per my current understanding, that can be managed by increasing a single measurement (point) time.

Overall, more discussion is needed to understand the exact needs for this setup.

TODO: Testing the new logic -- fast flush ends when pressure is halfway to the target (if operations are fast enough)
conclusion: fast flush also needs a choke OR it must be controlled by state duration time (must be short enough)
Q: Should GasX state have pump on or off...

## Measuring n spectra

In measurement loop 12th step of the sequence
(follow the "Measure" button), wrap "Measure"
and "Measure w/ GasX" buttons into the case structure which would trigger repeating
measurements for n times. Kind of a for loop,
but not really.

How would work:

* set n
* current n
* start_offset = 300

* measure n: False
|__do regular measurement
   |__tune: False
      |__measure->False
      measure: False
      |__tune->False

* measure n: True
|__current n == set n
   |__change file name to '...{Mn}'
   |__regular measurement
   |__clear plot
   |__measure n == False
   current n < set n
   |__change file name to '...{Mn}'
   |__regular measurement
   |__offset = start_offset
   |__clear plot
   |__current n += 1

Automation is tested on non-GasX measurement.
TODO: test on "Measure w/ GasX" (tested, works)

slow evac is triggered correctly now (double check)

Added a tunable parameter, which defines the pressure where
fast flush stops.

Super fast (2 ms) loop wait does not allow stopping fast 
flush sooner than ~30 mbar before  target pressure.

1563 (0x61B) (112 mV) peak for single point testing.

Changed cont measure data and path default

!!!For single measurement, tune step=0 !!!

Added Gasx state string to fast sensor state trigger.

Trying to delay measurement data recording trigger
(state string) by introducing timeout before sending out
the state string.
Still problems with measurement.

Fixed timeout
Added timeout for pump in gasx.
For each measurement pressure still need to adjust gasx
parameters (fast pump stop at n mbar to target etc.)
```
</details>

## 2025-05-23

**Summary:** A series of parameters (amplitude, phase) were systematically tested and recorded for different modulation frequencies (20 Hz to 300 Hz) and pressures (300, 600, 900 mbar). The main goal was to optimize these parameters for various experimental conditions.

<details>
<summary>Original Notes</summary>

```
23.05.2025
TODO:
for each freq:
	1) check change in peak intenstity for diff pressures
	2) adjust phase if necessary
	3) adjust amplitude

20, 32 ... 300 (10 mod freq values total)

10 measurements for each freq.

optimize modulation amplitude for pressure levels (and phase

initial parameters
* phase optimization:
	DFB phase 1 = -pi/2=0x9131
	DFB phase 2 = 0(2pi)=0x5131
* offset=0x41A for first H2O peak at 13.186 kOhm (19 degC)

PARAMS LIST:

mod frew = old (new/floored)
mod freq = 20 [Hz]
300 mbar: amp = 0x12, ph1 = 0x7FC9, ph2 = 0xBFC9 DONE
600 mbar: amp = 0x24, ph1 = 0x7701, ph2 = 0xB701 DONE
900 mbar: amp = 0x32, ph1 = 0x73F0, ph2 = 0xB3F0 DONE

mod freq = 32 (30) [Hz] (some interference..?)
300 mbar: amp = 0x12, ph1 = 0x6469, ph2 = 0xA469 DONE
600 mbar: amp = 0x24, ph1 = 0x5DF1, ph2 = 0x9DF1 DONE
900 mbar: amp = 0x32, ph1 = 0x5BDA, ph2 = 0x9BDA DONE

mod freq = 42 (40) [Hz]
300 mbar: amp = 0x12, ph1 = 0x4E29, ph2 = 0x8E29 DONE
600 mbar: amp = 0x24, ph1 = 0x4951, ph2 = 0x8951 DONE
900 mbar: amp = 0x32, ph1 = 0x4821, ph2 = 0x8821 DONE

# 0-line unstable here (ph1)
mod freq = 56 [Hz]
300 mbar: amp = 0x48, ph1 = 0x43B9, ph2 = 0x83B9
600 mbar: amp = 0x
900 mbar: amp

mod freq = 84 (80) [Hz] !!! Make sure this is in freq list  ^^!!!
300 mbar: amp = 0x12, ph1 = 0x8691, ph2 = 0x4691 DONE
600 mbar: amp = 0x24, ph1 = 0x8371, ph2 = 0x4371 DONE
900 mbar: amp = 0x32, ph1 = 0x8331, ph2 = 0x4331 DONE

mod freq = 98 (90) [Hz]
300 mbar: amp = 0x12, ph1 = 0x7469 , ph2 = 0x3469 DONE
600 mbar: amp = 0x24, ph1 = 0x7311, ph2 = 0x3311 DONE 
900 mbar: amp = 0x32, ph1 = 0x72D1, ph2 = 0x32D1 DONE

mod freq = 130 [Hz]
300 mbar: amp = 0x12, ph1 = 0x32F9, ph2 = 0xF2F9 DONE
600 mbar: amp = 0x24, ph1 = 0x32C1, ph2 = 0xF2C1 DONE
900 mbar: amp = 0x32, ph1 = 0x3281, ph2 = 0xF281 DONE

mod freq = 171 (170) [Hz]
300 mbar: amp = 0x12, ph1 = 0x72F9, ph2 = 0xB2F9 DONE
600 mbar: amp = 0x24, ph1 = 0x7349 , ph2 = 0xB349 DONE
900 mbar: amp = 0x32, ph1 = 0x7349 , ph2 = 0xB349 DONE

mod freq = 227 (220) [Hz]
300 mbar: amp = 0x12, ph1 = 0xA179, ph2 = 0x6179 DONE
600 mbar: amp = 0x24, ph1 = 0xA359, ph2 = 0x6359 DONE
900 mbar: amp = 0x32, ph1 = 0xA419, ph2 = 0x6419 DONE

mod freq = 300 [Hz]
300 mbar: amp = 0x12, ph1 = 0xB2C9, ph2 = 0x72C9 DONE
600 mbar: amp = 0x24, ph1 = 0xB739, ph2 = 0x7739 DONE
900 mbar: amp = 0x32, ph1 = 0xBF39, ph2 = 0x7F39 DONE



TODO: 
0) before making loop, do a few spectral measurements
to see mod amplitude effect. How too low or too 
high mod amplitude changes spectrum. DONE
1) add phase array control DONE
2) test close range amplitudes for one freq
3) do some full auto measurements for multiple frequencies. DONE
4) after auto, trigger "clear chart" button, add amplitude table and add amplitude setting to "initial values" button. DONE
5) water system stability DONE
	* water first (3 meas. for each freq.)
6) amplitude find
	* 20 hz  (300, 600, 900)

TODO
1) plotting peak height and height/trough ratio DONE
2) 600 mbar auto amplitude w/ faster gasx (10 sec.?) DONE
```
</details>

## 2025-06-17

**Summary:** The plan was to switch to a lower temperature and investigate the results. A problem was discovered where measurements were invalid because the water supply had been turned off, requiring a redo of previous experiments.

<details>
<summary>Original Notes</summary>

```
17.06.2025
TODO
1) Switch temperature to lower
2) see whats up
Also changed init amplitudes and added init values trigger after single measurements

Need to redo the measurments bc water
supply was turned off :D :///

REMINDER: full voltage offset range is 0x300 to 0x852 in LabView
REMINDER: offset1 values for a line at 19degC were ~ from 0x352 to 0x4B0
```
</details>

## 2025-06-18

**Summary:** The experiment was moved to a different spectral line at a higher temperature (22 degC). The plan was to conduct multiple amplitude tests for various pressures and discuss the results.

<details>
<summary>Original Notes</summary>

```
18.06.2025

1) Move to the other line (6983.6678 cm-1 [offset 0x578 to 0x640]) T_LD ~ 22 degC DONE
2) Do multiple amplitude:
	* expt (300 600 900 mbar)
	* sim (300 600 900 mbar
3) discuss further
```
</details>

## 2025-06-30

**Summary:** A mechanical chopper was connected for diagnostic purposes. New settings were defined for measurements using the chopper, including setting DFB Amplitude to zero for non-phase-sensitive measurements.

<details>
<summary>Original Notes</summary>

```
30.06.2025.
Connected mechanical chopper for diagnostics;
For measurements w/ chopper settings:
	* DFB Amplitude 0x0000
	* DFB Harm 1
	* DFB measamp 1 (non phase sensitive measurement)

TODO: Measurements for 300 600 900 mbar 22 degC offset range 0x300 0x852 DONE
```
</details>

## 2025-07-01

**Summary:** Analysis was performed on spectral line width, including Doppler vs. pressure analysis. The accuracy of the analysis was noted as uncertain.

<details>
<summary>Original Notes</summary>

```
01.07.2025.
Still line width mystery is not completely solved, but some measurements and simulations are done.
Some doppler vs pressure analysis is also done, although not certainly accurate
```
</details>

## 2025-07-02

**Summary:** The focus was on converting laser current (mA) to LabView offsets and then to wavenumbers. This was part of an effort to determine the effective laser linewidth at zero-amplitude modulation.

<details>
<summary>Original Notes</summary>

```
02.07.2025.

Need to convert from current [mA]
to offsets (decimal/hex labview values)
to wavenumbers, to determine what is the '0-amplitude-modulation'
determine effective laser linewidth

________
copmuter

white
green
none
none

cell
---------
```
</details>

## 2025-07-03

**Summary:** An attempt was made to use a PicoScope Arbitrary Waveform Generator (AWG) within LabView to control the laser current, replacing the existing DAQ control.

<details>
<summary>Original Notes</summary>

```
03.07.2025.

Try to use pico AWG in labview instead of DAQ, to control laser current 
```
</details>

## 2025-07-04

**Summary:** The PicoScope AWG was integrated into an existing Pico subVI in LabView. The main challenge was synchronizing the offset changes with the measurement loop, which has a 1-second duration due to sampling requirements.

<details>
<summary>Original Notes</summary>

```
04.07.2025.

1) Added AWG inside already existing Pico subVI.

Description:

I need to change offset on every 'single' measurement, to achieve effective scan over spectrum.
Sequence would be

Change offset -> measure -> change offset -> measure ...

Problem is, that current pico loop is 1.00 [s] long, due to the necessary sampling requirements.

So need to find, where and how exactly the main program progresses through the 'single' measurements

'single' measurement
------------------------------------------------------------------------
  AWG setting							     
----------------1[s]----------------
                                      actual measuring
				    ----------------1[s]----------------
```
</details>

## 2025-07-14

**Summary:** A monitoring option was added to observe the actual voltage being supplied to the laser, confirming it was within the expected range. The next step was to connect this signal to the laser input, but it appeared to be noisier than the original signal.

<details>
<summary>Original Notes</summary>

```
14.07.2025.
Need to insert monitoring option from CEPAS system to see what is the actual voltage that's being fed
into the laser. DONE (did with probes, w/o special soldered thing)

V_in is on the same order as expected!

Got to connect the signal to laser input. But that signal now seems even noisier than the gasera control board supplied one. So must find some source of noise or introduce some crazy filter!
```
</details>

## 2025-07-17

**Summary:** The main challenge was identifying the precise trigger point for measurements to ensure synchronization between the control system and the laser. Brute-force solutions were considered, such as stopping after each measurement or using a short sampling window, but these had significant drawbacks.

<details>
<summary>Original Notes</summary>

```
17.07.2025.
Need to find where exactly measurement gets triggered. We cannot wait for just a value change of control, we need to wait for the actual changed value which we should somehow see from LDTC1020...
PROBLEM: labview speaks to picoscope and CEPAS board separately, no direct connection in between them.
BRUTE FORCE SOLUTION 1: make measurement stop after each single measurement
BRUTE FORCE SOLUTION 2: make timebase/sampling window temporally short (DRAWBACK: unreliable power meas.)
QUESTION: how would pico loop work, if timebase left default, and just change the loop time to shorter
ANSWER: probably breaks something...(not tested)
```
</details>

## 2025-07-29

**Summary:** An attempt to use the PicoScope voltage in the original 'value changed?' condition failed. The focus remains on finding a proper way to synchronize the measurement trigger, with a brute-force method of long measurement time and a short PicoScope sample window being considered.

<details>
<summary>Original Notes</summary>

```
29.07.2025

will try to put pico voltage into OG 'value chgd?' condition.
Did not work. Trying to revert and figure out proper way.

TODO: Still need to figure out.
TODO: try brute force -- long measurement time and short pico sample window

dir: 35RH_picoSignal
```
</details>

## 2025-07-30

**Summary:** A wait time was added to the LabView loop following discussions, and the method for power and signal level measurements was updated to use a built-in signal processing function. Tests were conducted while reviewing the previous day's measurements.

<details>
<summary>Original Notes</summary>

```
30.07.2025

Added wait time after discussions w/ Juho. Changed the way power and signal level measurements are done in labview: now using built-in signal processing thing, instead of manually calculating something...
Doing tests and looking at yesterdays measurements...

dir: picoSignal_newLoop
```
</details>

## 2025-07-31

**Summary:** A critical error was discovered where the modulation frequency was set incorrectly (80 in hex instead of 80 in decimal). Measurements for 300, 600, and 900 mbar were redone with the corrected frequency.

<details>
<summary>Original Notes</summary>

```
31.07.2025
Frequency was WRONG!!! Changed to proper 80(84) Hz decimal (it was 80 in hex before!)

Redoing measurements for 300, 600, 900 mbar DONE
```
</details>

## 2025-08-05

**Summary:** A cable assembly was being prepared to connect to a filter, indicating a hardware modification to the experimental setup.

<details>
<summary>Original Notes</summary>

```
05.08.2025
Doing the cable assembly to connect to the filter
```
</details>

## 2025-08-06

**Summary:** After installing a filter, the phase and amplitude parameters were recalibrated for the filtered signal at 300, 600, and 900 mbar. A measurement at 900 mbar had to be redone due to an incorrect amplitude setting.

<details>
<summary>Original Notes</summary>

```
06.08.2025
Redo the phases and amplitudes for filtered signal

300 mbar DONE
600 mbar DONE
900 mbar DONE

redo 900 mbar measurment and do 600 mbar measurment. did 900 w/ wrong amplitude DONE
```
</details>

## 2025-08-08

**Summary:** The frequency sweep functionality was tested and confirmed to be working correctly. The functionality from the "Noise Test" window was also successfully integrated into the main "Laser" window, though some issues with USB/SPI commands remained.

<details>
<summary>Original Notes</summary>

```
08.08.2025

Test again how freq sweep is functioning (should work just fine!) DONE (works)
spectrum measure from 0x0300 to 1675 in decimal


TODO: Add functionality from "Legacy -> Noise Test" to the "Legacy -> Laser" window
Noise test add
still problems with opening device in 19.08.2025. Otherwise looks like implanted it in correct place...
Need to figure out when and how to call usbspi commands properly!
DONE
```
</details>

## 2025-08-20

**Summary:** The noise test functionality is now accessible through the main program, confirming the integration work from the previous entries. The new laser continues to work as expected.

<details>
<summary>Original Notes</summary>

```
20.08.2025
See above, noise test possible through main program. New laser still works.
```
</details>

## 2025-08-21

**Summary:** It was observed that the acoustic spectrum only shows rounded frequency values. This led to the suspicion that the frequency parameter only accepts integers rounded to the nearest ten.

<details>
<summary>Original Notes</summary>

```
21.08.2025
When looking at the acoustic spectrum, we can only see the rounded (floored) frequencies, so the suspicion is that we can pass only rounded numbers to the frequency parameter, rounded to 1e+1
```
</details>

## 2025-08-22

**Summary:** A series of 17 automated measurements were planned and executed at 600 mbar pressure and 130 Hz frequency. During this time, work began on writing Python scripts for data analysis.

<details>
<summary>Original Notes</summary>

```
22.08.2025
TODO:
Continue the measurements with 
freq 130
amp 24
press 600
meas time 1 s
auto n = 17

WHILE MEASURING:
Write scripts for data analysis (see notes in office!)

need to redo 600mbar n=17 (wrong phase)
DONE
```
</details>

## 2025-09-01

**Summary:** Noise measurements were taken at flat regions of the spectrum for various pressures and frequencies. A table of phase and noise values was recorded, with some measurements completed and others marked as to-do.

<details>
<summary>Original Notes</summary>

```
01.09.2025

Get noise at single points of spectrum (flat regions).

300 mbar  1      2	600 mbar  1	 2    	900 mbar  1	 2
7fc9 bfc9 4.7e-1 3.6e-5 7701 b701 4.5e-5 4.1e-5 73f0 b3f0 4.1e-5 4.5e-5
6469 a469 3.6e-4 2.2e-5 605a a05a 1.1e-4 3.0e-5 5df1 9df1 5.5e-5 2.8e-5
4e29 8e29 5.6e-5 1.6e-5 4951 8951 4.0e-5 2.0e-5 4821 8821 2.6e-5 2.9e-5
8691 4691 2.2e-5 1.3e-5 8371 4371 2.0e-5 1.8e-5 8331 4331 2.3e-5 1.4e-5
7469 3469 2.7e-5 1.6e-5 7311 3311 1.7e-5 1.9e-5 72d1 32d1 1.4e-5 2.0e-5
32f9 f2f9 1.7e-4 1.6e-5 32c1 f2c1 4.9e-4 1.2e-5 3281 f281 1.1e-4 1.5e-5
72f9 b2f9 1.8e-5 1.7e-5 7349 b349 1.5e-5 1.5e-5 7349 b349 1.2e-5 1.1e-5
a179 6179 2.3e-5 3.0e-5 a359 6359 1.6e-5 1.8e-5 a419 6419 1.9e-5 1.3e-5
b2c9 72c9 1.7e-5 2.1e-5 b739 7739 2.5e-5 2.4e-5 bf39 7f39 4.2e-5 4.1e-5
DONE 1		  	DONE 1	    	  	DONE 1
TODO 2		  	TODO 2	    	  	TODO 2
```
</details>

## 2025-09-11

**Summary:** A test plate with a groove was attached to the lift and aligned with the laser. All spectral and single-point noise measurement benchmarks were redone for all pressures and frequencies.

<details>
<summary>Original Notes</summary>

```
11.09.2025

Have attached the test plate with the groove to the lift.
DONE: Align so the laser goes through and screw the lift to the table
DONE: REDO all the spectral measurement benchmarks (all pressures and frequencies)
TODO: REDO all single measurement noise benchmarks
DONE: Record noise spectrum for all pressures (average of N=10)
```
</details>

## 2025-09-15

**Summary:** An additional piece of foam was added to the setup. This was likely done to correct the angle of the cell or to provide better vibration damping.

<details>
<summary>Original Notes</summary>

```
15.09.2025
Added an additional foam piece to correct the angle of cell
```
</details>

## 2025-09-22

**Summary:** This entry contains notes on tracking different cantilevers. The currently installed cantilever is identified as the 'r6c9 soi2'.

<details>
<summary>Original Notes</summary>

```
22.09.2025
CANTILEVER TRACKING

INSIDE : OLD <30u mass no.
Old one is inside 'r6c9 soi2' bottom right corner
```
</details>

## 2025-09-24

**Summary:** Different cantilevers were tested and compared. The old cantilever was reinstalled to check for leaks, and several new ones (R6C7, R6C5, R10C5) were tested for their noise characteristics and response to environmental sounds, with foam being used for damping.

<details>
<summary>Original Notes</summary>

```
24.09.2025
Put back in old cantilever to see if it still has some leak like new one... 

Old cantilever reinstalled: reacts to voice in amplitude range 500u-1000m, some low-f noise visible more than with new CL

Lets see the new one again: flatter spectrum overall, external voice like acoustics can easier hit above 1m signal levels. Much less low-f noise

Trying R6C7: Foam helps for dampening environment sounds. Noise high level is ~150u-250u. At 600 Hz it starts to fall, and at 800 Hz
it is already fallen to around 50u or less

Trying R6C5: With avg N=1 situation is similar  as with other two new CLs. with N=10 noise floor rises to around 100-150u, and noise delta looks like some 75u, but reduction in that is expected as sqrt(N).

Old: Looks like expected.

R10C5: looks something like between new 30u and old 30u(?) cantilevers.

TODO: look at the paper plan DONE

R10C5: Looks better but low freq noise floor still a bit elevated, see saved spectrum. Testing with absorption signal.
```
</details>

## 2025-09-25

**Summary:** The foam's shape was checked after settling overnight, and noise measurements were taken at 900 mbar. Interferogram data was saved for the old cantilever, and the r6c5 30u cantilever was tested for its Signal-to-Noise Ratio (SNR).

<details>
<summary>Original Notes</summary>

```
25.09.2025
Going by the plan
1) Checking how foam has reshaped overnight, checking noise at 900 mbar DONE
2) r5c10 60u signal 0.23, noise 4.64e-4 DONE
3) saving interferogram from both DSPs for OLD cantilever 2/2 DONE
4) afterwards try for r6c5 30u spectrum, check SNR and foam stand noise N5 DONE (everything 2x lower...)
```
</details>

## 2025-09-26

**Summary:** The interferometer alignment was checked and confirmed to be okay. A full frequency measurement was performed for the r6c5_soi2 30u cantilever, with a reminder to manually change the phase for the first measurement at 30 Hz.

<details>
<summary>Original Notes</summary>

```
26.09.2025
Interferometer alignment check DONE =OK
Will do all freq measurement for r6c5_soi2 30u cantilever

DONE
Make SURE that starting from 30Hz (n=3), change the PHASE for first one manually!!! AND PARAMETER SWEEP ON FOR FREQUENCY!!!!

try 60u cantilever from other wafer r5c10_soi1! (see below)
```
</details>

## 2025-09-29

**Summary:** A series of tests were conducted on the r5c10_soi1 cantilever. These included recording interferograms and noise data on both a standard stand and a foam bench, followed by a full frequency sweep and a single-point noise test.

<details>
<summary>Original Notes</summary>

```
29.09.2025
TODO: r5c10_soi1: DONE
	1) Interferogram stand DONE
	2) Noise stand DONE
	3) interferogram foam bench DONE
	4) Noise foam bench DONE
	5) Full frequency sweep @ 900 mbar DONE
	6) single point noise test at three frequencies...(for this cantilever only, then decide what to do next) DONE
	7) USED OLD3 cantilever in the end DONE
```
</details>

## 2025-10-02

**Summary:** A full benchmark, including frequency sweeps and noise tests, was completed for the R6C5 SOI2 30u cantilever at 300, 600, and 900 mbar. Similar tests were started for the R4C5 SOI1 150u and R1C4 SOI1 500u cantilevers.

<details>
<summary>Original Notes</summary>

```
02.10.2025
TODO: Do FULL benchmark for:
	1) R6C5 SOI2 30u:DONE
		900:
			freq sweep DONE
			noise DONE
		600:
			freq sweep DONE
			noise DONE
		300:
			freq sweep DONE
			noise DONE 

	2) R4C5 SOI1 150u: (adjusted mirrors back to power meter pico_ch_A = 0.0347)
		900:
			freq sweep DONE
			noise DONE

	3) R1C4 SOI1 500u
		900:
			bad interferogram DONE
```
</details>

## 2025-10-07

**Summary:** A frequency sweep was performed for the R8C6 soi2 150u cantilever with an attached mass. It was noted that the phase adjustment at 40 Hz was coincidentally similar to the original cantilever's 20 Hz adjustment.

<details>
<summary>Original Notes</summary>

```
07.10.2025
Doing 40, 70, 120 Hz sweep for R8C6 soi2 150u cantilever w/ mass DONE
Lucky coincidence: 40 Hz phase adjustment very similar to the original OLD cantilever 20Hz adjustment
```
</details>

## 2025-10-08

**Summary:** The task for the day was to extract information from the new cantilevers that had been tested. This likely involves analyzing the data collected in the previous sessions.

<details>
<summary>Original Notes</summary>

```
08.10.2025
Need to extract info from the new cantilevers!
```
</details>

## 2025-10-10

**Summary:** The 60u cantilever with an attached mass (R5C3) was tested. This required adjusting the mirrors due to reflections, and a frequency sweep was performed at 60, 70, and 80 Hz where a signal was possibly detected.

<details>
<summary>Original Notes</summary>

```
10.10.2025
Testing 60u with mass R5C3
Some reflection again, needed to adjust both mirrors to power meter P=0.025 @ current=3E8
chosen 60 70 80 Hz freq sweep (at least some kind of signal there possibly...DONE
single noise DONE
```
</details>

## 2025-10-16

**Summary:** After a discussion about the cantilevers, the plan was to set up a rubber damping system to observe its effect on noise. Measurements were planned for the R5C3 cantilever at 20, 30, and 40 Hz, keeping in mind the second harmonic.

<details>
<summary>Original Notes</summary>

```
16.10.2025
Discussing cantilevers

Do rubber setup to see how affects the noise

R5C3 do 20(40), 30(60), 40(80) Hz measurement!! Reminded about 2ND harmonic!!!
```
</details>

## 2025-10-24

**Summary:** The cantilevers were prepared to be sent away. The original old cantilever was put back into the system, but the interferometer had not yet been readjusted.

<details>
<summary>Original Notes</summary>

```
24.10.2025
Will send away the cantilevers;
put back inside old one but not yet readjusted interferometer or anything, just had it put inside
```
</details>

## 2025-10-27

**Summary:** The interferometer was readjusted for the original old cantilever that was reinstalled in the previous session.

<details>
<summary>Original Notes</summary>

```
27.10.2025
Adjusted interferogram for now for old og cantilever
```
</details>

## 2025-10-29

**Summary:** A backup of the n_gasx routine was created for the LabView test program. The necessary components were copied into LabView, and the next step is to conduct a multi-frequency test for the open-cell setup.

<details>
<summary>Original Notes</summary>

```
29.10.2025
DONE: Make backup for n_gasx routine for labview test prog. (filename "KT008_TH11 test progrm_backup_n_gasx.vi")
DONE: Copied in labview all necessary things. Left to test.
TODO: multi-freq test for open-cell cell.
```
</details>

## 2025-10-30

**Summary:** With TED200C set to 11.448 kOhm, the chosen range for the frequency sweep is 1420-1510 in LabView decimal units. The pressure is atmospheric but will be recorded as 900 mbar in filenames. The possibility of adding `README.md` files to the root and subdirectories for better navigation and understanding, with Gemini deducing data types based on commit dates and `notes.txt`, was also considered. A notebook for the open-cell initial test was created.

<details>
<summary>Original Notes</summary>

```
30.10.2025
With TED200C set to 11.448 kOhm, chosen range is 1420-1510 in labview decimal units. Doing the freq sweep for the line in that
'offsets' range. Pressure is atmospheric but in filenames it will be 900 mbar!! Note that! Could add the overview README.md in root
directory and also in subdirectories, could try to let gemini deduce what kind of data is there in each of those, based on last
commit dates and `notes.txt` file (this file). Note old range for current offsets in hex are (3E8 - 6D6). For the line will use
1420-1510 (decimal)
Created the notebook for opencell cell initial test.
```
</details>

## 2025-10-31

**Summary:** A frequency sweep from 20-190 Hz (1st harmonic) was performed on the open-cell setup with a Helmholtz filter to observe its effects. The results suggest the filter was effective, particularly around 130 Hz (260 Hz 2nd harmonic).

<details>
<summary>Original Notes</summary>

```
31.10.2025
Doing the freq sweep 20-190 for oc cell with Helmholtz filter, to see the effects, first harmonic 20-190 Hz with steps of 10
(for 2nd harmonic it means 40-380 w/ steps of 20 Hz). Idk, for me it seems like the filter worked, especially if the frequency for it was somewhere around 130 Hz (260 Hz in barcharts looks the best).
```
</details>