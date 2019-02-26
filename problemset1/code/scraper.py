# @Author: Magnus Berg Sletfjerding <mag>
# @Date:   2019-02-07T00:06:55+01:00
# @Email:  mbs@chem.ku.dk
# @Project: improved-eureka
# @Filename: scraper.py
# @Last modified by:   mag
# @Last modified time: 2019-02-12T22:13:45+01:00



#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import scrapy
import pandas as pd

# urls
urls_to_scrape = ["https://kenpom.com/index.php?y={}".format(year) for year in list(range(2002, 2020))]

## Make spider class
class BasketSpider(scrapy.Spider):
    """docstring for BasketSpider."""
    name = "basket_spider"
    start_urls = urls_to_scrape
    # start_urls = [
    # "https://kenpom.com/index.php?y=2014",
    # ]

    def parse(self, response):
        year  = str(response)[-5:-1]
        # # First get col names
        # col_names = []
        #
        # table_head = response.xpath('//*[@class="thead2"]')
        # col_names_container = table_head[0].xpath('th//a')

        ## INSTEAD of scraping Row names (redundant and unnecessary),
        ## define manually
        col_names = ['Rk', 'Team', '_', 'Sd', 'Conf', 'W-L', 'AdjEM', 'AdjO', 'AdjORk', 'AdjD', 'AdjDRk', 'AdjT', 'AdjTRk', 'Luck', 'LuckRk', 'SOS', 'SOSRk', 'OppO', 'OppORk', 'OppD', 'OppDRk', 'NCSOS', 'NCSOSRk']



        # Find the data for each row
        ## table found by '//*[@id="ratings-table"]//tbody'
        ## rows are '//*[@id="ratings-table"]//tbody//tr'
        rows = response.xpath('//*[@id="ratings-table"]//tbody//tr')
        for row in rows:
            ## Some rows have number None, let's ignore those
            if row.xpath('td[1]//text()').extract_first() != None:
                row_elements = row.xpath('td//text()').getall()
                ## Some teams are not seeded, so two conditions for our dict
                col_names_temp = col_names[:2] + col_names[4:]

                if len(row_elements) == len(col_names):
                    row_dict = dict(zip(col_names,row_elements))
                else:
                    row_dict = dict(zip(col_names_temp,row_elements))

                ## Make dict to DF, input
                if row_dict['Rk'] == '1':
                    df = pd.DataFrame(data=row_dict, columns = row_dict.keys(), index=[row_dict['Rk'],] )
                else:
                    df_temp = pd.DataFrame(data=row_dict, columns = row_dict.keys(), index=[row_dict['Rk'],] )
                    df = pd.concat([df, df_temp])

        # remove empty
        df.drop('_', axis=1, inplace=True)
        # Save DF
        df.to_csv(
        '../data/basket_data_{}.csv'.format(year)
        )

        df.to_pickle(
        path = '../data/basket_data_{}.pkl'.format(year)
        )

        pass
