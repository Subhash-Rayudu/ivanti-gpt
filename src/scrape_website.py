import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

def get_page_text(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    return soup.get_text()


def get_links(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    
    links = []
    for link in soup.find_all('a'):
        href = link.get('href')
        
        if not href or (not href.startswith('/') and 'https://www.ivanti.com' not in href) or\
        any(sub in href for sub in ('/en-au/', '/en-gb/', '/de/', '/es/', '/fr/', '/it/', '.cn', '/ja/', 
                                    '/trials/', '/lp/', '/email-protection', '/webinars/', '/demo-videos', 
                                    '/free-trials', '/doc/', '/rss', '/careers/', '/blog/', '/promo/')):
            # print("skipping:", href)
            continue
        # print(href)
        # if href:
        if href and href.startswith('/'):
            full_url = urljoin(url, href)
            links.append(full_url)
        else:
            links.append(href)
    # print(links)
    return links



def scrape_website(base_url):
    visited_links = set()
    to_visit = [base_url]
    cnt = 0
    while to_visit:
        current_url = to_visit.pop(0)
        if current_url not in visited_links:
            visited_links.add(current_url)
            print(f"Visiting: {current_url}")
            try:
                page_text = get_page_text(current_url)
                if(page_text == None):
                    continue
                # print(page_text)
                links = get_links(current_url)
                # TODO: Add the list of linking pages to the visited page to have relevance in model
                # document = page_text + links
                to_visit.extend(link for link in links if link not in visited_links)
                cnt += 1
            except Exception as e:
                print(f"Failed to process {current_url}: {e}")
        # print("cnt:", cnt)
    print("cnt: ", len(visited_links), "\nall links: ", sorted(visited_links))

scrape_website('https://www.ivanti.com/')