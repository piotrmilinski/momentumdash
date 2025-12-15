# Momentum Dash

A lightweight Streamlit dashboard that visualizes weekly trend-following positioning across a small ETF universe.

## Setup

1. Create and activate a virtual environment
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Windows: .venv\Scripts\activate
   ```
2. Install dependencies
   ```bash
   pip install -r requirements.txt
   ```
3. Download price history (cached to `data/prices_daily.csv`)
   ```bash
   python update_data.py
   ```
4. Launch the app
   ```bash
   streamlit run app.py
   ```

## One-click starts (Windows)
- Create a `start_app.bat` file with:
  ```bat
  call .venv\Scripts\activate
  streamlit run app.py
  ```
- Create a `refresh_data.bat` file with:
  ```bat
  call .venv\Scripts\activate
  python update_data.py
  ```

## Notes
- The app caches data locally in the `data/` directory (ignored by git).
- Use the "Refresh prices now" button inside the app to update data on demand.
