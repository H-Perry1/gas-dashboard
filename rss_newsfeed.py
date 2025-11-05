# rss_feed.py
import streamlit as st
from datetime import datetime, timezone, timedelta
import feedparser
import requests
from dateutil import parser as date_parser

def display_rss_feed(feeds, keywords):
    """Display live filtered RSS feed content from the last 24 hours."""
    all_news = []
    now = datetime.now(timezone.utc)
    cutoff_time = now - timedelta(hours=24)

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
    }

    for feed_url in feeds:
        if not feed_url.strip():
            continue  # skip blanks

        try:
            response = requests.get(feed_url, headers=headers, timeout=20)
            response.raise_for_status()
            feed = feedparser.parse(response.content)

            for entry in feed.entries:
                # Try to parse publication date
                published_dt = None
                if "published" in entry:
                    try:
                        published_dt = date_parser.parse(entry.published)
                        if not published_dt.tzinfo:
                            published_dt = published_dt.replace(tzinfo=timezone.utc)
                    except Exception:
                        published_dt = None

                # Skip if too old
                if published_dt and published_dt < cutoff_time:
                    continue

                # Keyword filtering
                content = (entry.title + " " + entry.get("summary", "")).lower()
                if any(keyword.lower() in content for keyword in keywords):
                    all_news.append({
                        "title": entry.title,
                        "link": entry.link,
                        "summary": entry.get("summary", "No summary"),
                        "published": entry.get("published", "Unknown time"),
                    })

        except Exception as e:
            st.warning(f"Error fetching {feed_url}: {e}")

    # Display results
    if not all_news:
        st.info("No matching news found in the last 24 hours.")
    else:
        for item in all_news[:10]:  # show top 10
            st.markdown(f"### [{item['title']}]({item['link']})")
            st.write(item["summary"])
            st.caption(f"Published: {item['published']}")
            st.markdown("---")
