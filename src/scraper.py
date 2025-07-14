import praw
from typing import List, Dict

def extract_username(url: str) -> str:
    """Extracts the username from a Reddit profile URL."""
    return url.split("/user/")[1].split("/")[0]

def scrape_user_data(reddit: praw.Reddit, username: str) -> tuple[List[Dict], List[Dict]]:
    """Scrapes posts and comments for a given Reddit username."""
    posts = []
    comments = []

    try:
        redditor = reddit.redditor(username)
        for submission in redditor.submissions.new(limit=10):  # Limited to 10 recent posts
            posts.append({"text": submission.title + "\n" + (submission.selftext if submission.selftext else ""), "url": submission.url})
        for comment in redditor.comments.new(limit=50):  # Limited to 50 recent comments
            comments.append({"text": comment.body, "url": f"https://www.reddit.com{comment.permalink}"})

    except Exception as e:
        print(f"Error scraping data for {username}: {str(e)}")

    print(f"Scraped {len(posts)} posts and {len(comments)} comments for {username}")
    for c in comments[:5]:  #printing first 5 comments for debugging
        print(f"Comment: {c['text'][:50]}... URL: {c['url']}")
    return posts, comments