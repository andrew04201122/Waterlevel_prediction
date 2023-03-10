def water_trans(water_data):
    water_data = water_data.replace('後平',"後平(国)")
    water_data = water_data.replace('矢口第一',"矢口第一(国)")
    water_data = water_data.replace('矢口第二',"矢口第二(国)")
    water_data = water_data.replace('神野瀬川',"神野瀬川(国)")
    water_data = water_data.replace('南畑敷',"南畑敷(国)")
    water_data = water_data.replace('神辺',"神辺(国)")
    water_data = water_data.replace('吉田',"吉田(国)")
    water_data = water_data.replace('宇津戸川',"宇津戸川(国)")
    water_data = water_data.replace('上庄',"上庄(国)")
    water_data = water_data.replace('府中',"府中(国)")
    water_data = water_data.replace('川井',"川井(国)")
    water_data = water_data.replace('計納',"計納(国)")
    water_data = water_data.replace('大津',"大津(国)")
    water_data = water_data.replace('竹の花',"竹の花(国)")
    water_data = water_data.replace('山守橋',"山守橋(国)")
    water_data = water_data.replace('古川',"古川(国)")
    water_data = water_data.replace('市場',"市場(国)")
    water_data = water_data.replace('上安田',"上安田(国)")
    water_data = water_data.replace('上原橋',"上原橋(国)")
    water_data = water_data.replace('矢野原',"矢野原(国)")
    water_data = water_data.replace('中深川',"中深川(国)")
    water_data = water_data.replace('粟屋',"粟屋(国)")
    water_data = water_data.replace('郷分',"郷分(国)")
    water_data = water_data.replace('上戸手',"上戸手(国)")
    water_data = water_data.replace('滝山',"滝山(国)")
    water_data = water_data.replace('玖村',"玖村(国)")
    water_data = water_data.replace('祇園大橋',"祇園大橋(国)")
    water_data = water_data.replace('庄原',"庄原(国)")
    water_data = water_data.replace('加計',"加計(国)")
    water_data = water_data.replace('黒滝',"黒滝(国)")
    water_data = water_data.replace('江波(旧太田川)',"江波(旧太田川)(国)")
    water_data = water_data.replace('草津',"草津(国)")
    water_data = water_data.replace('西神島',"西神島(国)")
    water_data = water_data.replace('下ヶ原',"下ヶ原(国)")
    water_data = water_data.replace('山手',"山手(国)")
    water_data = water_data.replace('三篠橋',"三篠橋(国)")
    water_data = water_data.replace('防鹿',"防鹿(国)")
    water_data = water_data.replace('和木(国)',"和木")
    water_data = water_data.replace('飯室',"飯室(国)")
    water_data = water_data.replace('山手左岸',"山手左岸(国)")
    water_data = water_data.replace('永野山',"永野山(国)")
    water_data = water_data.replace('矢多田川',"矢多田川(国)")
    water_data = water_data.replace('湯来',"湯来(国)")
    water_data = water_data.replace('三次',"三次(国)")
    water_data = water_data.replace('土居',"土居(国)")
    water_data = water_data.replace('長和久',"長和久(国)")
    water_data = water_data.replace('新川橋',"新川橋(国)")
    water_data = water_data.replace('尾関山',"尾関山(国)")
    water_data = water_data.replace('下土師',"下土師(国)")
    water_data = water_data.replace('新市',"新市(国)")
    water_data = water_data.replace('御幸',"御幸(国)")
    return water_data

def rain_trans(rain_data):
    rain_data = rain_data.replace(dict_rs)
    return rain_data

def tide_trans(tide_data):
    tide_data = tide_data.replace({"倉橋漁港":"倉橋港","呉阿賀港":"呉(阿賀)港", "柿浦漁港":"柿浦港"})
    return tide_data

def ws_order():
    return ws_list_order
def rs_order():
    return rs_list_order
def ts_order():
    return ts_list_order

dict_rs = {'三入':"三入(気)",'呉市蒲刈':"呉市蒲刈(気)",'甲田':"甲田(気)",'土師':"土師(国)",  
          '弥栄ダム':"弥栄ダム(国)",'神辺':"神辺(国)",'津名':"津名(国)",'上奥原':"上奥原(国)",
           '王泊':"王泊(国)",'藤原':"藤原(国)",'矢草(国)':"矢草北",'矢草':"矢草南",'倉橋':"倉橋(気)",
           '廿日市津田':"廿日市津田(気)",'新市(電)':"新市",'雄鹿原':"雄鹿原(国)",'時安':"時安(国)",
           '七曲':"七曲(国)",'大林':"大林(国)",'宇津戸':"宇津戸(国)",'油木':"油木(国)",'箕島':"箕島(国)",
           '西城':"西城(国)",'狩留家':"狩留家(国)",'府中':"府中(国)",'向原':"向原(国)",'生口島':"生口島(気)",
           '上下':"上下(気)",'大津':"大津(国)",'八幡(電)':"八幡(気)",'南原':"南原(国)",'筒賀':"筒賀(国)",
           '美土里':"美土里(国)",'上安田':"上安田(国)",'東城':"東城(国)",'大竹':"大竹(気)",'楢原':"楢原(国)",
           '河内(電)':"河内",'中道':"中道(国)",'大谷':"大谷(国)",'上領家':"上領家(国)",'総領':"総領(国)",
           '広島':"広島(国)",'灰塚ダム':"灰塚ダム(国)",'大月':"大月(国)",'大暮':"大暮(国)",'八幡':"八幡(気)",
           '安宿':"安宿(気)",'庄原':"庄原(気)",'田万里(西堀坂)':"田万里",'津田':"津田(国)",'吉舎':"吉舎(国)",
           '加計':"加計(国)",'都志見':"都志見(気)",'溝口':"溝口(国)",'布野':"布野(国)",'君田':"君田(気)",
           '本地':"本地(国)",'大朝':"大朝(国)",'高蓋':"高蓋(国)",'志路原':"志路原(国)",'内黒山':"内黒山(気)",
           '本郷':"本郷(気)",'佐伯湯来':"佐伯湯来(気)",'松原':"松原(国)",'飯室':"飯室(国)",'道後山':"道後山(気)",
           '八田原':"八田原(国)",'志和':"志和(気)",'温井ダム':"温井ダム(国)",'高瀬':"高瀬(国)",'東広島':"東広島(気)",
           '高宮':"高宮(国)",'世羅':"世羅(気)",'戸山':"戸山(国)",'白木(三日市)':"白木",'湯来':"湯来(国)",
           '三次':"三次(国)",'駅家': "駅家(国)",'福山':"福山(国)",'大谷山':"大谷山(国)",'板木':"板木(国)",
           '黒目':"黒目(国)",'鈴張':"鈴張(国)"
          }  

used_ws = ['多治比', '南原', '向原', '上甲立', '上安', '石原', '岡ノ下', '大須', '亀山', '三入南', '瀬野', '下原', '門田', '菅沢', '石内', '呉地', '向田', '久地', '町田', '二河', '惣引谷', '宮内', '平良', '水ノ越', '石井谷', '春木', '大朝', '失平', '風早', '新庄', '下野', '河戸', '中島', '樋ノ詰', '松ヶ瀬', '下見', '三津', '竹原', '古河', '御薗宇', '駅家中島', '瀬戸山北', '西中条', '中野', '福田', '万能倉', '上安井', '大黒', '今津', '大橋', '新市宮内', '沼隈', '山野', '古市', '府中砂川', '津之郷', '松永', '手城', '服部', '二森', '西宮', '柳井橋', '沼田東', '南方', '本郷(三原)', '菅川橋', '美之郷', '中之町', '椋梨', '十日市', '下志和地', '岡田', '三玉', '小田幸', '上壱', '和知', '小文', '藤兼', '下布野', '西城', '東城', '戸郷川', '比和', '高', '本郷(廿日市)', '釜ヶ原', '岩倉', '白川', '小深川', '中地', '市原', '和木', '奥条', '中河内', '船木', '七宝', '甲山', '駅前', '前原', '七社', '大谷池', '百谷', '沼', '加茂', '種', '出雲', '丸門田', '市', '伊尾', '青近', '上中', '高井', '今田', '三篠橋(国)', '江波(旧太田川)(国)', '古川(国)', '上原橋(国)', '新川橋(国)', '白木(国)', '中深川(国)', '上庄(国)', '湯来(国)', '土居(国)', '加計(国)', '飯室(国)', '中野(国)', '玖村(国)', '矢口第二(国)', '矢口第一(国)', '長和久(国)', '祇園大橋(国)', '草津(国)', '黒滝(国)', '滝山(国)', '後平(国)', '下ヶ原(国)', '防鹿(国)', '山手(国)', '府中(国)', '郷分(国)', '上戸手(国)', '矢野原(国)', '御幸(国)', '神辺(国)', '西神島(国)', '新市(国)', '山手左岸(国)', '伊尾(国)', '永野山(国)', '山守橋(国)', '宇津戸川(国)', '矢多田川(国)', '粟屋(国)', '尾関山(国)', '大津(国)', '計納(国)', '南畑敷(国)', '神野瀬川(国)', '三次(国)', '庄原(国)', '吉田(国)', '竹の花(国)', '上安田(国)', '市場(国)', '川井(国)', '下土師(国)']


ws_list_order= ['多治比', '南原', '向原', '上甲立', '上安', '石原', '岡ノ下', '大須', '亀山', '三入南', '瀬野', '下原', '門田', '菅沢', '石内', '呉地', '向田', '久地', '町田', '二河', '惣引谷', '宮内', '平良', '水ノ越', '石井谷', '春木', '大朝', '失平', '風早', '新庄', '下野', '河戸', '中島', '樋ノ詰', '松ヶ瀬', '下見', '三津', '竹原', '古河', '御薗宇', '駅家中島', '瀬戸山北', '西中条', '中野', '福田', '万能倉', '上安井', '大黒', '今津', '大橋', '新市宮内', '沼隈', '山野', '古市', '府中砂川', '津之郷', '松永', '手城', '服部', '二森', '西宮', '柳井橋', '沼田東', '南方', '本郷(三原)', '菅川橋', '美之郷', '中之町', '椋梨', '十日市', '下志和地', '岡田', '三玉', '小田幸', '上壱', '和知', '小文', '藤兼', '下布野', '西城', '東城', '戸郷川', '比和', '高', '本郷(廿日市)', '釜ヶ原', '岩倉', '白川', '小深川', '中地', '市原', '和木', '奥条', '中河内', '船木', '七宝', '甲山', '駅前', '前原', '七社', '大谷池', '百谷', '沼', '加茂', '種', '出雲', '丸門田', '市', '伊尾', '青近', '上中', '高井', '今田', '三篠橋(国)', '江波(旧太田川)(国)', '古川(国)', '上原橋(国)', '新川橋(国)', '白木(国)', '中深川(国)', '上庄(国)', '湯来(国)', '土居(国)', '加計(国)', '飯室(国)', '中野(国)', '玖村(国)', '矢口第二(国)', '矢口第一(国)', '長和久(国)', '祇園大橋(国)', '草津(国)', '黒滝(国)', '滝山(国)', '後平(国)', '下ヶ原(国)', '防鹿(国)', '山手(国)', '府中(国)', '郷分(国)', '上戸手(国)', '矢野原(国)', '御幸(国)', '神辺(国)', '西神島(国)', '新市(国)', '山手左岸(国)', '伊尾(国)', '永野山(国)', '山守橋(国)', '宇津戸川(国)', '矢多田川(国)', '粟屋(国)', '尾関山(国)', '大津(国)', '計納(国)', '南畑敷(国)', '神野瀬川(国)', '三次(国)', '庄原(国)', '吉田(国)', '竹の花(国)', '上安田(国)', '市場(国)', '川井(国)', '下土師(国)']
rs_list_order = ['西部建設', '上瀬野', '熊野町', '江波', '福木', '中山新町', '楠那', '己斐', '堂免橋', '日浦', '上原', '揚倉山', '海田', '坂', '彩が丘', '川根', '下甲立', '吉田町', '八千代町', '美土里町', '白木', '多治比', '佐々部', '向原坂', '桑田', '奥畑', '五月が丘', '五日市観音', '井口台', '牛田早稲田', '祇園山本', '大柿町', '秋月', '中町', '菅沢', '瀬戸内ハイツ', '杉並台', '高祖', '蒲刈大浦', '豊島', '内海', '呉支所', '蒲刈町', '焼山', '郷原', '呉', '広', '小坪', '仁方', '警固屋', '波多見', '宇和木', '田戸', '下蒲刈', '大長', '川尻', '斎島', '尾曽郷', '室尾', '天応', '田原', '原', '廿日市支所', '大野', '吉和', '馬の口', '栗谷', '大竹市', '宮島町', '浅原', '友和', '佐伯', '玖島', '安芸太田支所', '芸北', '杉ノ泊', '江河内', '水谷', '黒峠', '猪山', '高野', '二川', '川小田', '大塚', '川戸', '新都', '中原', '吉木', '中ノ原', '布原', '新庄', '黒瀬町', '河内', '高美が丘', '郷曽', '久芳', '吉原', '田万里', '小梨', '東広島支所', '志和東', '下三永', '吉川', '篠', '三津', '大崎町', '上組', '岩伏', '明石', '竹原', '仁賀ダム', '東部建設', '上安井', '加茂', '沼隈町', '井関', '瀬戸', '南松永', '羽高', '大浦', '田尻', '山野', '古市', '二森', '神石町', '川南', '油木安田', '下豊松', '梶山田', '三原支所', '甲原', '本谷', '菅川橋', '高尾', '美之郷', '因島', '和木', '西野', '末光', '外浦', '林', '吉田', '有井', '黒川', '安田', '別迫', '野間川ダム', '北部建設', '上壱', '三次石原', '南畑敷', '青河', '櫃田', '東入君', '横谷', 'ゆめランド', '敷地', '仁賀', '作木西野', '甲奴本郷', '竹地谷', '永田', '中領家', '庄原支所', '川北', '戸郷川', '本村町', '西城中野', '比和', '高暮', '新市', '川東', '小瀬川ダム', '栗栖', '魚切ダム', '重光', '後畑', '野呂川ダム', '椋梨ダム', '下徳良', '造賀', '乃美', '福富ダム', '甲山', '三川', '賀茂', '七社', '四川', '御調', '江木', '山田川ダム', '梶毛ダム', '庄原ダム', '八坂', '中山', '矢草北', '矢草南', '奴メリ谷', '戸山(国)', '大林(国)', '向原(国)', '白木(国)', '狩留家(国)', '湯来(国)', '大谷(国)', '七曲(国)', '加計(国)', '飯室(国)', '高瀬(国)', '広島(国)', '溝口(国)', '筒賀(国)', '南原(国)', '鈴張(国)', '楢原(国)', '松原(国)', '雄鹿原(国)', '上奥原(国)', '大暮(国)', '王泊(国)', '温井ダム(国)', '津田(国)', '中道(国)', '弥栄ダム(国)', '府中(国)', '福山(国)', '箕島(国)', '大谷山(国)', '御調(国)', '神辺(国)', '駅家(国)', '賀茂(国)', '八田原(国)', '宇津戸(国)', '高蓋(国)', '古城(国)', '大津(国)', '吉田(国)', '高暮(国)', '美土里(国)', '高宮(国)', '西城(国)', '庄原(国)', '総領(国)', '吉舎(国)', '津名(国)', '三次(国)', '大月(国)', '板木(国)', '比和(国)', '布野(国)', '油木(国)', '志路原(国)', '西野(国)', '上安田(国)', '上領家(国)', '黒目(国)', '灰塚ダム(国)', '大朝(国)', '藤原(国)', '本地(国)', '土師(国)', '東城(国)', '時安(国)', '広島(気)', '三入(気)', '佐伯湯来(気)', '大竹(気)', '廿日市津田(気)', '呉(気)', '倉橋(気)', '呉市蒲刈(気)', '竹原(気)', '志和(気)', '東広島(気)', '安宿(気)', '本郷(気)', '生口島(気)', '福山(気)', '上下(気)', '府中(気)', '世羅(気)', '油木(気)', '君田(気)', '三次(気)', '高野(気)', '庄原(気)', '東城(気)', '道後山(気)', '美土里(気)', '甲田(気)', '加計(気)', '内黒山(気)', '都志見(気)', '王泊(気)', '八幡(気)', '大朝(気)']
ts_list_order = ['大竹港', '広島港', '柿浦港', '呉(阿賀)港', '倉橋港', '御手洗港', '竹原港', '木江港', '糸崎港', '尾道港', '土生港', '横田港', '福山港']