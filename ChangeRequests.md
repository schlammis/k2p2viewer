# Change Requests & Open Questions

## Features

- **Allan deviation tab**
  - Done, but not included in uncertainty budget yet
- **ppm/µg checkbox** for mass plot right axis
    - Done
- **g/mg checkbox** for left axis and the values displayed on the left (ref mass, measured mass)
  - Done
- **Legend** for blue band and red band (overall uncertainty vs. Type A uncertainty)
  - Done
- **Make the left tab copy-pastable in Excel**
  - right click on the table or header will bring up context menu that will allow you to copy to clipboard
  

## Bugs / Investigations

- **Environmental empty**: Sometimes the Attocube does not correct the index unless enabled in their UI,
  and sometimes it does. Consider showing an error dialog or greying out the result when Environmental data is empty.
  - Done. Check it thoroughly. I could not find a run that had all 0.

- **Red uncertainty band keeps expanding**: Investigate why the red band grows over time.

- **3 ppm discrepancy**: Figure out the discrepancy between k2p2 and the KA program.

## Questions

- **Conventional mass in Report tab**: Both conventional and true mass values appear to read the same.
  Is the conventional mass being corrected to 20 °C, air density 1.2 kg/m³, and material density 8 000 kg/m³?

- **Exclude outlier**: Is the exclude-outlier option working correctly?
  -  Yes — uses a Huber robust estimator with a 5-sigma cut. Requires more than 6 points.

- **Type A uncertainty**: Is Type A reported as std. dev. or std. uncertainty (i.e. divided by √n)?
