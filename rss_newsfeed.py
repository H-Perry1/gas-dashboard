# rss_feed.py
import streamlit as st
from datetime import datetime
import feedparser
import requests

def display_rss_feed(feeds, keywords):
    """Display live filtered RSS feed content."""
    all_news = []

    for feed_url in feeds:
        if not feed_url.strip():
            continue  # skip blanks
        try:
            response = requests.get(feed_url, timeout=10)
            response.raise_for_status()
            feed = feedparser.parse(response.content)

            for entry in feed.entries:
                content = (entry.title + " " + entry.get("summary", "")).lower()
                if any(keyword.lower() in content for keyword in keywords):
                    all_news.append({
                        "title": entry.title,
                        "link": entry.link,
                        "summary": entry.get("summary", "No summary"),
                        "published": entry.get("published", datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
                    })

        except Exception as e:
            st.warning(f"Error fetching {feed_url}: {e}")

    if not all_news:
        st.info("No matching news found right now.")
    else:
        for item in all_news[:10]:  # show top 10
            st.subheader(item["title"])
            st.markdown(f"[Read more]({item['link']})")
            st.write(item["summary"])
            st.caption(f"Published: {item['published']}")
            st.markdown("---")
