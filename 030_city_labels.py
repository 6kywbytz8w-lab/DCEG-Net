CITY_NAME_EN = {'长春': 'Changchun', '吉林': 'Jilin', '四平': 'Siping', '辽源': 'Liaoyuan', '通化': 'Tonghua', '白山': 'Baishan', '松原': 'Songyuan', '白城': 'Baicheng', '延边': 'Yanbian'}

def city_en(name):
    return CITY_NAME_EN.get(name, name)

def city_list_en(city_list):
    return [city_en(c) for c in city_list]

def rename_city_df(df):
    df2 = df.copy()
    if isinstance(df2.index, pd.Index):
        df2.index = [city_en(x) for x in df2.index]
    if isinstance(df2.columns, pd.Index):
        df2.columns = [city_en(x) for x in df2.columns]
    return df2

city_order_en = city_list_en(city_order) if 'city_order' in globals() else None
