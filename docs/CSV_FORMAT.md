# LibreView CSV Format

The first data source for GlucoPilot is a CSV export from LibreView.

The application will merge glucose values coming from historical readings and sensor scans into a single GlucoseReading entity.

The first version will import:

- Timestamp
- Glucose value in mg/dL
- Rapid insulin units
- Basal insulin units