The crawler supports three modes:
	•	accept — accept all cookies
	•	reject — reject all cookies
	•	block — block third-party trackers using Disconnect’s blocklist

It also records:
	•	HAR files
	•	Full-page screenshots (before & after consent)
	•	Screen-capture videos
	•	document.cookie values

All output is written into:
	•	crawl_data_accept/
	•	crawl_data_reject/
	•	crawl_data_block/

⸻

1. Requirements

Python Version
Python 3.8 or higher
Install Python packages
Run inside your project folder:
pip install -r requirements.txt

Your requirements.txt contains only:

playwright>=1.40
Install Playwright browser binaries
After installing the Python package, run:
python -m playwright install

This downloads Chromium, which the crawler uses.

⸻

2. Input File Format (sites-list.csv)

The crawler expects a CSV file containing:
URL, country

Example:

https://www.nytimes.com,us
https://www.nypost.com,us
example.com,nl

A scheme (https://) is optional — the crawler adds it when needed.

⸻

3. Running the Crawler

All commands run from the same directory as crawl.py.
Accept mode (accept all cookies)
python crawl.py -m accept -l sites-list.csv
Reject mode (reject all cookies)
python crawl.py -m reject -l sites-list.csv
Block mode (block trackers)
Block mode requires a Disconnect blocklist JSON file.

Example (using your downloaded file):

python crawl.py -m block -l sites-list.csv --disconnect ./disconnect.json
Or using the file in crawler_src/:
python crawl.py -m block -l sites-list.csv --disconnect crawler_src/services.json

⸻

4. Output Files

For each site (example: example.com), the crawler produces:
Screenshots
example.com_pre_consent.png
example.com_post_consent.png
HAR
example.com.har
Video
example.com.webm
document.cookie dump
example.com_document_cookie.json


⸻

5. Troubleshooting

ModuleNotFoundError for har_utils
Add to your notebook:

import sys
sys.path.append("/Users/kaghuvahanreddy/Documents/OTP/crawler_src")
Block mode requires –disconnect

Always pass:

--disconnect path/to/services.json
Stuck on some sites

Use:

--debug --no-video --no-har --site-timeout 60
and inspect _pre_consent.png to adjust dictWords.py.

⸻

6. Contact / Notes

This crawler implementation satisfies:
	•	GDPR cookie consent handling (accept/reject)
	•	Tracker blocking using Disconnect categories
	•	Automated scrolling to bypass bot detection
	•	HAR + full screenshot + video capture
	•	Consistent waits
	•	region-safe navigation handling

