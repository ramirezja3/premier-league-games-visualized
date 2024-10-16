import csv
import scrapy
game_list = list()
dicts_list = list()

class MatchesSpider(scrapy.Spider):
    name = "matches"
    allowed_domains = ["www.skysports.com"]
    start_urls = ["https://www.skysports.com/premier-league-results/2023-24"]

    def start_requests(self):
        urls = ["https://www.skysports.com/premier-league-results/2023-24", "https://www.skysports.com/premier-league-results/2022-23", 
                "https://www.skysports.com/premier-league-results/2021-22", "https://www.skysports.com/premier-league-results/2020-21",
                "https://www.skysports.com/premier-league-results/2019-20", "https://www.skysports.com/premier-league-results/2018-19"]
        for i in urls:
            yield scrapy.Request(url=i, callback=self.parse_games)

    def parse_games(self, response):
        for i in response.css('div div div div div div a::attr(href)').extract():
            if i[:5] == 'https' and i[-1:].isnumeric():
                parts = i.split('/')
                parts.insert(-1, "stats")
                i = '/'.join(parts)
                game_list.append(i)
        for url in game_list:
            yield scrapy.Request(url=url, callback=self.parse_items)

    def parse_items(self, response):
        attributes_team1 = {}
        attributes_team2 = {}

        attributes_team1['Team'] = response.css('div div h4 span span::text').getall()[0]
        attributes_team1['Score'] = response.css('div div h4 span span::text').getall()[1]
        attributes_team1['Status'] = 'Home'
        attributes_team1['Possession Percentage'] = response.css('div div div div div div div span span::text').getall()[0]
        attributes_team1['Total Shots'] = response.css('div div div div div div div span span::text').getall()[2]
        attributes_team1['Shots on Target'] = response.css('div div div div div div div span span::text').getall()[4]
        attributes_team1['Passing Percentage'] = response.css('div div div div div div div span span::text').getall()[10]
        attributes_team1['Tackles Percentage'] = response.css('div div div div div div div span span::text').getall()[18]
        attributes_team1['Fouls Committed'] = response.css('div div div div div div div span span::text').getall()[24]
        attributes_team1['Fouls Won'] = response.css('div div div div div div div span span::text').getall()[26]
        attributes_team1['Yellow Cards'] = response.css('div div div div div div div span span::text').getall()[28]
        attributes_team1['Red Cards'] = response.css('div div div div div div div span span::text').getall()[30]

        if int(response.css('div div h4 span span::text').getall()[1]) > int(response.css('div div h4 span span::text').getall()[3]):
            attributes_team1['Result'] = 'Win'
        elif int(response.css('div div h4 span span::text').getall()[1]) < int(response.css('div div h4 span span::text').getall()[3]):
            attributes_team1['Result'] = 'Lose'
        else:
            attributes_team1['Result'] = 'Draw'
        attributes_team1['Year'] = ((response.css('div div div div p time::text').get()).split()[-1]).strip('.')
        if (response.css('div div div div p span::text').getall()[0])[-1:] == '.':    
            attributes_team1['Stadium'] = (response.css('div div div div p span::text').getall()[0]).strip('.')
        else:
            attributes_team1['Stadium'] = response.css('div div div div p span::text').getall()[0]

        try:
            attributes_team1['Attendance'] = (response.css('div div div div p span::text').getall()[3]).strip('.')
        except IndexError:
            attributes_team1['Attendance'] = "0 (Covid-19)"



        attributes_team2['Team'] = response.css('div div h4 span span::text').getall()[2]
        attributes_team2['Score'] = response.css('div div h4 span span::text').getall()[3]
        attributes_team2['Status'] = 'Away'
        attributes_team2['Possession Percentage'] = response.css('div div div div div div div span span::text').getall()[1]
        attributes_team2['Total Shots'] = response.css('div div div div div div div span span::text').getall()[3]
        attributes_team2['Shots on Target'] = response.css('div div div div div div div span span::text').getall()[5]
        attributes_team2['Passing Percentage'] = response.css('div div div div div div div span span::text').getall()[11]
        attributes_team2['Tackles Percentage'] = response.css('div div div div div div div span span::text').getall()[19]
        attributes_team2['Fouls Committed'] = response.css('div div div div div div div span span::text').getall()[25]
        attributes_team2['Fouls Won'] = response.css('div div div div div div div span span::text').getall()[27]
        attributes_team2['Yellow Cards'] = response.css('div div div div div div div span span::text').getall()[29]
        attributes_team2['Red Cards'] = response.css('div div div div div div div span span::text').getall()[31]

        if int(response.css('div div h4 span span::text').getall()[1]) < int(response.css('div div h4 span span::text').getall()[3]):
            attributes_team2['Result'] = 'Win'
        elif int(response.css('div div h4 span span::text').getall()[1]) > int(response.css('div div h4 span span::text').getall()[3]):
            attributes_team2['Result'] = 'Lose'
        else:
            attributes_team2['Result'] = 'Draw'
        attributes_team2['Year'] = ((response.css('div div div div p time::text').get()).split()[-1]).strip('.')
        if (response.css('div div div div p span::text').getall()[0])[-1:] == '.':    
            attributes_team2['Stadium'] = (response.css('div div div div p span::text').getall()[0]).strip('.')
        else:
            attributes_team2['Stadium'] = response.css('div div div div p span::text').getall()[0]
        try:
            attributes_team2['Attendance'] = (response.css('div div div div p span::text').getall()[3]).strip('.')
        except IndexError:
            attributes_team2['Attendance'] = "0 (Covid-19)"

        if '\n' not in response.css('div div div div div div div span span::text').getall()[1]:
            dicts_list.append(attributes_team1)
            dicts_list.append(attributes_team2)

        with open('games.csv', 'w', newline='') as csvfile:
            fieldnames = dicts_list[0].keys()
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for i in dicts_list:
                writer.writerow(i)

