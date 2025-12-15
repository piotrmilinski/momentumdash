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
- If you encounter SSL certificate errors when downloading prices (e.g. behind a
  corporate proxy), set `YAHOO_VERIFY_SSL=false` in your environment before
  running `python update_data.py` or launching the app to disable certificate
  verification.

## How to resolve a GitHub merge conflict (web UI)
If GitHub shows the conflict editor like in the screenshot, you can finish the
merge directly in the browser:

1. In the left sidebar, click each conflicting file (e.g., `README.md`) to open
   the conflict view.
2. Inside the file, choose the version you want to keep by deleting the
   conflict markers that look like:
   ```
   <<<<<<< HEAD
   your version
   =======
   incoming change
   >>>>>>> branch-name
   ```
   Keep only the desired content and remove the marker lines.
3. Repeat for every conflicting file until no markers remain, then click
   **Mark as resolved**.
4. Once all files are resolved, click **Commit merge** (or **Commit changes**).
5. Finally, click the green **Merge pull request** button to finish.

Tip: if you're unsure which version to keep, copy the two versions into a text
editor to combine them before pasting the final text back into GitHub.
