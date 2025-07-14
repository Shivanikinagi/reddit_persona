Reddit Persona Generator
This repository contains a script to generate user personas based on Reddit activity.
Setup

Install dependencies:pip install -r requirements.txt


Create a .env file in the project root directory with the following content:REDDIT_CLIENT_ID=YourClientID
REDDIT_CLIENT_SECRET=YourClientSecret

Obtain these credentials from https://www.reddit.com/prefs/apps.
Create an output directory in the project folder:mkdir output



Usage

Run the script:python main.py


Enter a Reddit user profile URL (e.g., https://www.reddit.com/user/kojied/).
Check the generated persona in output\<username>_persona.txt.

Files

scraper.py: Scrapes Reddit posts and comments (renamed from reddit_persona.py based on your import).
persona_generator.py: Builds the user persona using a transformer model with fallback.
main.py: Orchestrates the process.
requirements.txt: Lists dependencies.
