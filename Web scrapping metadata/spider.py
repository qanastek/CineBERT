import scrapy
import re

from tutorial.items import TutorialItem

class SpiderSpider(scrapy.Spider):
    name = 'spider'
    allowed_domains = ['allocine.fr']
    start_urls = ['https://www.allocine.fr/film/fichefilm_gen_cfilm=241016.html']

    # Every page
    def parse(self, response):
        id_films = open("list_movies_ids.txt", "r")

        for id in id_films:
            id = "/film/fichefilm_gen_cfilm=" + str(id) + ".html"
            url = response.urljoin(id)
            yield scrapy.Request(url=url, callback=self.parse_item)

    def parse_item(self, response):
        # initialize item
        item = TutorialItem()
        try:
            # get title , realisateur , id of movie
            title = response.css('.titlebar-title::text').extract_first()
            realisateur = set(response.css('.meta-body-direction .blue-link::text').extract())
            id = response.css('main::attr(data-seance-geoloc-redir)').extract_first()

            # get duree
            duree = response.css('.meta-body-info').extract_first().split('<span class="spacer">/</span>')[1].replace('\n', '')
            if "span" in duree:
                duree = ""

            # get notes
            notes = response.css('.stareval-note::text').extract()
            avg_note_presse = notes[0]
            avg_note_spectateurs = notes[1]

            # get categoris
            categories = []
            datas =  response.xpath("//div[@class='meta-body-item meta-body-info']//text()").extract()[2:]
            for c in datas:
                if '/' not in c and '\n' not in c:
                    categories.append(c)

            acteurs = []
            datas = response.xpath("//div[@class='meta-body-item meta-body-actor']//text()").extract()[2:]
            for c in datas:
                if '/' not in c and '\n' not in c:
                    acteurs.append(c)
            # 
            item['id'] = id
            item['title'] = title
            item['duree'] = duree
            item['realisateur'] = realisateur
            item['avg_note_presse'] = avg_note_presse
            item['avg_note_spectateurs'] = avg_note_spectateurs
            item['categories'] = categories
            item['acteurs'] = acteurs

        except Exception as e:
            item = []
        yield item
