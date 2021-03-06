from urllib import request
from bs4 import BeautifulSoup
import requests
import re
import asyncio
import aiohttp
import os
import time


# 获取网页源代码
def get_html(url):
    html = request.urlopen(url=url, timeout=10).read().decode('UTF-8')
    soup = BeautifulSoup(html, features='lxml')
    return soup


# 获取页数和项目数
def get_pagenum(soup):
    divPages = soup.find("div", {"class": "pages"})
    str_divPages = str(divPages)
    getPage = re.search(r".*getPageNav\((\d+)\, (\d+)\, (\d+)\, .*", str_divPages)
    pageNum = int(getPage.group(1))     # 页数
    itemNum = int(getPage.group(3))     # 项目数
    return pageNum, itemNum


# 获取页面上每张图片所在的网址（返回list）
def get_pic_url(soup):
    urlBase = "http://www.namoc.org/zsjs/gczp/cpjxs/"
    imgUrlRow = soup.find_all("a", {"href": re.compile(".*\.htm.*"), "target": "_blank"})
    imgUrl = [request.urljoin(urlBase, u["href"][1:]) for u in imgUrlRow]
    return imgUrl


# 获取每张图片的信息（名字、作者、年份、图片源网址）并下载
async def get_pic(imgUrlSingle, foldername, pagenum):        # 此imgUrl为str，代表单张大图网址
    soup = get_html(imgUrlSingle)
    imgimg = soup.find("img", {"id": "mimg", "src": re.compile(".*\.(jpg|png)")})
    url = imgimg["src"]
    type = re.search(r".*(\.jpg|\.png)", url).group(1)
    imginfo = soup.find("div", {"class": "info"})
    imgname = imginfo.find("h2")
    name = str(imgname.contents[0])
    # name = name.replace("/", " ")
    # name = name.replace("\"", " ")
    # name = name.replace(":", " ")
    authorname = "不详"
    times = "不详"
    for info in imginfo.find_all("p"):
        infotmp = str(info.contents[0])
        authortmp = re.search(r".*作者.(.+)", infotmp)
        timetmp = re.search(r".*年[代份].(.+)", infotmp)
        if authortmp is not None:
            authorname = authortmp.group(1)
            # authorname = authorname.replace("/", " ")
        if timetmp is not None:
            times = timetmp.group(1)
    if len(name) != 0:
        # 记录图片信息
        recordpath = "./img/record.txt"
        filetxt = open(recordpath, 'a+')
        itmnum = len(open(recordpath, 'r').readlines())
        itmnum += 1
        if itmnum == 1:
            itmtitle = "序号\t来源\t类型\t页数\t作品名\t作者\t年代"
            filetxt.write(itmtitle + "\r")
        itminfo = str(itmnum) + "\t" +"中国美术馆\t" + foldername + "\t" + str(pagenum + 1) + "\t" + name + "\t"\
                     + authorname + "\t" + times
        filetxt.write(itminfo + "\r")
        filetxt.close()
        # 保存图片
        filename = str(itmnum) + ".jpg"
        prtfilename = name + "_" + authorname + "_" + times + type
        r = requests.get(url)
        with open('./img/%s/%s' % (foldername, filename), 'wb') as f:
            f.write(r.content)
            print(prtfilename)
        # response = await session.get(url)
        # with open('./img/%s/%s' % (foldername, filename), 'wb') as f:
        #     print(type(response))
        #     await f.write(response.content)
    return url, name, authorname, times


# 处理（下载）该页所有图片
async def page_crawl(loop, pagenum, url, foldername):       # url放urlNow
    if pagenum == 0:
        urltmp = url + "index.htm"
    else:
        urltmp = url + "index_" + str(pagenum) + ".htm"
    soup = get_html(urltmp)
    urllist = get_pic_url(soup)
    tasks = [loop.create_task(get_pic(urllist[i], foldername, pagenum)) for i in range(0, len(urllist))]
    await asyncio.wait(tasks)


# 本体
async def main(loop):
    urlBase = "http://www.namoc.org/zsjs/gczp/cpjxs/"
    pcDic = {"youhua": "yh/", "zhongguohua": "zgh/"}
    pcNameList = list(pcDic.keys())
    for i in range(0, len(pcDic)):
        # 创建文件夹
        path = "./img/" + pcNameList[i] + "/"
        os.makedirs(path, exist_ok=True)
        # 解析品类网页
        urlNow = urlBase + pcDic[pcNameList[i]]
        soupNow = get_html(urlNow)
        pageNum = get_pagenum(soupNow)[0]       # 转url时记得减一
        itemNum = get_pagenum(soupNow)[1]
        print(pcNameList[i], "pageNum: ", pageNum, "itemNum: ", itemNum)
        # 进入品类不同页操作
        for j in range(0, pageNum):
            await page_crawl(loop, j, urlNow, pcNameList[i])


t0 = time.time()
loop = asyncio.get_event_loop()
loop.run_until_complete(main(loop))
loop.close()
print("Async total time:", time.time() - t0)