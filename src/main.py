from dotenv import load_dotenv
import praw
import os
from scraper import extract_username, scrape_user_data
from persona_generator import build_persona_with_huggingface

def main():
    load_dotenv()

    reddit = praw.Reddit(
        client_id=os.getenv('REDDIT_CLIENT_ID'),
        client_secret=os.getenv('REDDIT_CLIENT_SECRET'),
        user_agent='RedditPersonaBuilder/1.0'
    )

    url = input("Enter the Reddit user profile URL (e.g., https://www.reddit.com/user/kojied/): ")
    username = extract_username(url)

    posts, comments = scrape_user_data(reddit, username)

    persona = build_persona_with_huggingface(username, posts, comments)
    print(persona)  # Optional

if __name__ == "__main__":
    main()