import pandas as pd
from joblib import load
from data_trans import water_trans, rain_trans, tide_trans,ws_order,ts_order,rs_order
from warnings import simplefilter
import datetime

simplefilter(action="ignore", category=pd.errors.PerformanceWarning)
class ScoringService(object):
    @classmethod
    def get_model(cls, model_path):
        """Get model method

        Args:
            model_path (str): Path to the trained model directory.

        Returns:
            bool: The return value. True for success.
        """
        cls.model = load(model_path + '/linear.joblib') 

        return True

    @classmethod
    def predict(cls, input): # 前日の水位をそのまま予測とするモデル
        """Predict method

        Args:
            input: meta data of the sample you want to make inference from (dict)

        Returns:
            dict: Inference for the given input.

        """
        tide =  pd.DataFrame()
        rain = pd.DataFrame()
        water = pd.DataFrame()
        start = datetime.datetime.now()
        tidedata = pd.DataFrame(input['tidelevel'])
        tidedata = tidedata.replace({'M':0.0, '*':0.0, '-':0.0, '--': 0.0, '**':0.0})
        tidedata = tidedata.fillna(0.0)

        tidedata1 = tide_trans(tidedata)
        tide_data_update = tidedata1.groupby('hour')
        end = datetime.datetime.now()
        print("tide執行時間：", end - start)
        
        start = datetime.datetime.now()
        raindata = pd.DataFrame(input['rainfall'])
        raindata = raindata.replace({'M':0.0, '*':0.0, '-':0.0, '--': 0.0, '**':0.0})
        raindata = raindata.fillna(0.0)
        raindata1 = rain_trans(raindata)
        rsdrop_list = []
        for index,row in raindata1.iterrows():
            if row['station'] not in used_rs:
                rsdrop_list.append(index)
        raindata1 = raindata1.drop(rsdrop_list)
        rain_data_update = raindata1.groupby('hour')
        end = datetime.datetime.now()
        print("rain執行時間：", end - start)
        
        start = datetime.datetime.now()
        waterdata = pd.DataFrame(input['waterlevel'])
        waterdata = waterdata.replace({'M':0.0, '*':0.0, '-':0.0, '--': 0.0, '**':0.0})
        waterdata = waterdata.fillna(0.0)
        waterdata1 = water_trans(waterdata)
        wsdrop_list = []
        for index,row in waterdata1.iterrows():
            if row['station'] not in used_ws:
                wsdrop_list.append(index)
        waterdata1 = waterdata1.drop(wsdrop_list)
        water_data_update = waterdata1.groupby('hour')
        end = datetime.datetime.now()
        print("water執行時間：", end - start)
        
        
        

        start = datetime.datetime.now()
        for i in range(24):
            tide = pd.concat([tide,tide_data_update.get_group(i).drop_duplicates(subset=["station"],keep="last")])
            rain = pd.concat([rain,rain_data_update.get_group(i).drop_duplicates(subset=["station"],keep="last")])
            water = pd.concat([water,water_data_update.get_group(i).drop_duplicates(subset=["station"],keep="last")])
        end = datetime.datetime.now()
        print("remove duplicate and concat執行時間：", end - start)
            
        start = datetime.datetime.now()
        rain1 = rain.pivot(index = 'hour',columns = 'station',values = 'value')
        tide1 = tide.pivot(index = 'hour',columns = 'station',values = 'value')
        water1 = water.pivot(index = 'hour',columns = 'station',values = 'value')
        end = datetime.datetime.now()
        print("pivot執行時間：", end - start)


        start = datetime.datetime.now()
        df_rs = pd.DataFrame()
        df_rs[rs_order()] = 0.0
        df_rs = pd.concat([df_rs,rain1],join='inner')
        print(f"df_rs.shape: {df_rs.shape}")
        print(f"rain column: {list(df_rs.columns)}")
        
        
        df_ws = pd.DataFrame()
        df_ws[ws_order()] = 0.0
        df_ws = pd.concat([df_ws,water1],join='inner')
        print(f"df_ws.shape: {df_ws.shape}")
        print(f"water column: {list(df_ws.columns)}")

        df_ts = pd.DataFrame()
        df_ts[ts_order()] = 0.0
        df_ts = pd.concat([df_ts,tide1],join='inner')
        print(f"df_ts.shape: {df_ts.shape}")
        print(f"tide column: {list(df_ts.columns)}")
        
        
        fullinput = pd.concat([df_ws,df_rs,df_ts],axis = 1)
        print(f'fullinput.shape: {fullinput.shape}')
        fullinput = fullinput.replace({'M':0.0, '*':0.0, '-':0.0, '--': 0.0, '**':0.0})
        fullinput = fullinput.fillna(0.0)
        fullinput = fullinput.astype(float)
        end = datetime.datetime.now()
        print("fulldata concat執行時間：", end - start)

        start = datetime.datetime.now()
        station_list = []
        time_list = list(range(0,24))
        return_time_list = []
        ws_list = ws_order()
        for i in range(len(ws_list)):
            station_list.extend([ws_list[i]]*24)
            return_time_list.extend(time_list)
        #print(f'time list: {return_time_list}')
        #print(f'ws list: {station_list}')
        
        myDataFrame = pd.DataFrame()
        myDataFrame['hour'] = return_time_list
        myDataFrame['station'] = station_list
        end = datetime.datetime.now()
        print("create hour and station執行時間：", end - start)

        start = datetime.datetime.now()
        pred = cls.model.predict(fullinput)
        myDataFrame['value'] = list(pred.flatten('F'))
        #print(myDataFrame)
        prediction = myDataFrame[['hour', 'station', 'value']].to_dict('records')
        end = datetime.datetime.now()
        print("predict執行時間：", end - start)


        return prediction



used_ws = ['多治比', '南原', '向原', '上甲立', '上安', '石原', '岡ノ下', '大須', '亀山', '三入南', '瀬野', '下原', '門田', '菅沢', '石内', '呉地', '向田', '久地', '町田', '二河', '惣引谷', '宮内', '平良', '水ノ越', '石井谷', '春木', '大朝', '失平', '風早', '新庄', '下野', '河戸', '中島', '樋ノ詰', '松ヶ瀬', '下見', '三津', '竹原', '古河', '御薗宇', '駅家中島', '瀬戸山北', '西中条', '中野', '福田', '万能倉', '上安井', '大黒', '今津', '大橋', '新市宮内', '沼隈', '山野', '古市', '府中砂川', '津之郷', '松永', '手城', '服部', '二森', '西宮', '柳井橋', '沼田東', '南方', '本郷(三原)', '菅川橋', '美之郷', '中之町', '椋梨', '十日市', '下志和地', '岡田', '三玉', '小田幸', '上壱', '和知', '小文', '藤兼', '下布野', '西城', '東城', '戸郷川', '比和', '高', '本郷(廿日市)', '釜ヶ原', '岩倉', '白川', '小深川', '中地', '市原', '和木', '奥条', '中河内', '船木', '七宝', '甲山', '駅前', '前原', '七社', '大谷池', '百谷', '沼', '加茂', '種', '出雲', '丸門田', '市', '伊尾', '青近', '上中', '高井', '今田', '三篠橋(国)', '江波(旧太田川)(国)', '古川(国)', '上原橋(国)', '新川橋(国)', '白木(国)', '中深川(国)', '上庄(国)', '湯来(国)', '土居(国)', '加計(国)', '飯室(国)', '中野(国)', '玖村(国)', '矢口第二(国)', '矢口第一(国)', '長和久(国)', '祇園大橋(国)', '草津(国)', '黒滝(国)', '滝山(国)', '後平(国)', '下ヶ原(国)', '防鹿(国)', '山手(国)', '府中(国)', '郷分(国)', '上戸手(国)', '矢野原(国)', '御幸(国)', '神辺(国)', '西神島(国)', '新市(国)', '山手左岸(国)', '伊尾(国)', '永野山(国)', '山守橋(国)', '宇津戸川(国)', '矢多田川(国)', '粟屋(国)', '尾関山(国)', '大津(国)', '計納(国)', '南畑敷(国)', '神野瀬川(国)', '三次(国)', '庄原(国)', '吉田(国)', '竹の花(国)', '上安田(国)', '市場(国)', '川井(国)', '下土師(国)']

used_rs = ['西部建設', '上瀬野', '熊野町', '江波', '福木', '中山新町', '楠那', '己斐', '堂免橋', '日浦', '上原', '揚倉山', '海田', '坂', '彩が丘', '川根', '下甲立', '吉田町', '八千代町', '美土里町', '白木', '多治比', '佐々部', '向原坂', '桑田', '奥畑', '五月が丘', '五日市観音', '井口台', '牛田早稲田', '祇園山本', '大柿町', '秋月', '中町', '菅沢', '瀬戸内ハイツ', '杉並台', '高祖', '蒲刈大浦', '豊島', '内海', '呉支所', '蒲刈町', '焼山', '郷原', '呉', '広', '小坪', '仁方', '警固屋', '波多見', '宇和木', '田戸', '下蒲刈', '大長', '川尻', '斎島', '尾曽郷', '室尾', '天応', '田原', '原', '廿日市支所', '大野', '吉和', '馬の口', '栗谷', '大竹市', '宮島町', '浅原', '友和', '佐伯', '玖島', '安芸太田支所', '芸北', '杉ノ泊', '江河内', '水谷', '黒峠', '猪山', '高野', '二川', '川小田', '大塚', '川戸', '新都', '中原', '吉木', '中ノ原', '布原', '新庄', '黒瀬町', '河内', '高美が丘', '郷曽', '久芳', '吉原', '田万里', '小梨', '東広島支所', '志和東', '下三永', '吉川', '篠', '三津', '大崎町', '上組', '岩伏', '明石', '竹原', '仁賀ダム', '東部建設', '上安井', '加茂', '沼隈町', '井関', '瀬戸', '南松永', '羽高', '大浦', '田尻', '山野', '古市', '二森', '神石町', '川南', '油木安田', '下豊松', '梶山田', '三原支所', '甲原', '本谷', '菅川橋', '高尾', '美之郷', '因島', '和木', '西野', '末光', '外浦', '林', '吉田', '有井', '黒川', '安田', '別迫', '野間川ダム', '北部建設', '上壱', '三次石原', '南畑敷', '青河', '櫃田', '東入君', '横谷', 'ゆめランド', '敷地', '仁賀', '作木西野', '甲奴本郷', '竹地谷', '永田', '中領家', '庄原支所', '川北', '戸郷川', '本村町', '西城中野', '比和', '高暮', '新市', '川東', '小瀬川ダム', '栗栖', '魚切ダム', '重光', '後畑', '野呂川ダム', '椋梨ダム', '下徳良', '造賀', '乃美', '福富ダム', '甲山', '三川', '賀茂', '七社', '四川', '御調', '江木', '山田川ダム', '梶毛ダム', '庄原ダム', '八坂', '中山', '矢草北', '矢草南', '奴メリ谷', '戸山(国)', '大林(国)', '向原(国)', '白木(国)', '狩留家(国)', '湯来(国)', '大谷(国)', '七曲(国)', '加計(国)', '飯室(国)', '高瀬(国)', '広島(国)', '溝口(国)', '筒賀(国)', '南原(国)', '鈴張(国)', '楢原(国)', '松原(国)', '雄鹿原(国)', '上奥原(国)', '大暮(国)', '王泊(国)', '温井ダム(国)', '津田(国)', '中道(国)', '弥栄ダム(国)', '府中(国)', '福山(国)', '箕島(国)', '大谷山(国)', '御調(国)', '神辺(国)', '駅家(国)', '賀茂(国)', '八田原(国)', '宇津戸(国)', '高蓋(国)', '古城(国)', '大津(国)', '吉田(国)', '高暮(国)', '美土里(国)', '高宮(国)', '西城(国)', '庄原(国)', '総領(国)', '吉舎(国)', '津名(国)', '三次(国)', '大月(国)', '板木(国)', '比和(国)', '布野(国)', '油木(国)', '志路原(国)', '西野(国)', '上安田(国)', '上領家(国)', '黒目(国)', '灰塚ダム(国)', '大朝(国)', '藤原(国)', '本地(国)', '土師(国)', '東城(国)', '時安(国)', '広島(気)', '三入(気)', '佐伯湯来(気)', '大竹(気)', '廿日市津田(気)', '呉(気)', '倉橋(気)', '呉市蒲刈(気)', '竹原(気)', '志和(気)', '東広島(気)', '安宿(気)', '本郷(気)', '生口島(気)', '福山(気)', '上下(気)', '府中(気)', '世羅(気)', '油木(気)', '君田(気)', '三次(気)', '高野(気)', '庄原(気)', '東城(気)', '道後山(気)', '美土里(気)', '甲田(気)', '加計(気)', '内黒山(気)', '都志見(気)', '王泊(気)', '八幡(気)', '大朝(気)']