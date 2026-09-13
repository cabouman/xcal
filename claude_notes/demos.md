# Demos planned for xcal 2.0

This is the planning list.  It covers every demo from xcal 1 and
every dataset we currently have.

## Simulation demos

| Demo | What it shows | xcal 1 equivalent | Status |
|---|---|---|---|
| demo_simulated_multi_voltage.py | Reflection tube at 3 voltages, 4 rods.  Recovers takeoff angle, filter, and scintillator. | demo_spec_est_3_voltages, tutorials T01 and T02 | Done |
| demo_simulated_multi_filtration.py | Synchrotron with 2 filtrations, 2 rods.  Recovers both filter thicknesses and the scintillator. | None (new; mirrors the ALS experiment) | Done |
| demo_simulated_transmission.py | Transmission tube (Versa style) at 3 voltages.  Recovers the target thickness from the Geant4 table. | Tutorial T03 | To build |

## Real data demos

| Demo | Data | xcal 1 equivalent | Status |
|---|---|---|---|
| demo_als_measured.py | ALS Beamline 8.3.2, 8 scans in hand (data/README.md) | demo_als | To build |

Not planned for 2.0: tutorial T04 (the analytical source model was
dropped) and the Zeiss Versa real-data demo (no data; Charlie will
ask Aditya Mohan).
