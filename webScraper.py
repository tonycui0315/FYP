from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
import sys
import csv

def get_page_links(pageDriver, baseUrl):
    a_tags = driver.find_elements_by_tag_name('a')

    # Get all urls
    urls = [tag.get_attribute('href') for tag in a_tags]

    #Get urls that start with baseUrl
    useful_urls = [url for url in urls if url and baseUrl in url]

    return useful_urls

def get_page_text(pageDriver):
    pageBody = pageDriver.find_element_by_xpath("/html/body")
    if not pageBody:
        return ""
    return pageBody.text


visited_urls = set()
unvisited_urls = set()
error_urls = set()

options = webdriver.ChromeOptions()
# without opening a Chrome tab
options.add_argument("--headless")
driver = webdriver.Chrome(executable_path=ChromeDriverManager().install(), options=options)

# fileName = input('Please enter file name to write: ')
baseUrl = input("Please enter website's base url: ")
unvisited_urls.add(baseUrl)

ct = 0
while unvisited_urls:
    # Remove url from unvisited and add it to visited
    currPage = unvisited_urls.pop()
    print(currPage)
    visited_urls.add(currPage)

    # Get page content
    try:
        driver.get(currPage)
    except:
        error_urls.add(currPage)
        print('Error getting page: ', currPage)
        continue

    # Get links
    try:
        currLinks = get_page_links(driver, baseUrl)
    except:
        error_urls.add(currPage)
        print('Error getting links: ', currPage)
        continue

    # Get text
    try:
        currText = get_page_text(driver)
        textToWrite = currText.splitlines()
        with open(f'testCSV.csv', 'w') as csvfile:
            fieldnames = ['id', 'text']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()
            for text in textToWrite:
                writer.writerow({'id': ct, 'text': {text}})
                ct += 1
    except:
        error_urls.add(currPage)
        print('Error getting text: ', currPage)
        continue

print("-----------------------------------------DATA RECOVER IS DONE")
sys.exit('Program finished')