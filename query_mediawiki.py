import requests

def search_mediawiki_tags(tags):
	url = 'http://en.wikipedia.org/w/api.php'
	data = '{"action":"query", "list":"search","srsearch": ' + tags + ', "srwhat":"text", "format":"json"}'
	response = requests.post(url, data=data, headers={"Content-Type": "application/json"})
	return response