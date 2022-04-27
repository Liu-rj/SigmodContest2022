import pandas as pd
import re

# brands = ['dell', 'lenovo', 'acer', 'asus', 'hp']
brands = ['compaq', 'toshiba', 'sony', 'ibm', 'epson', 'xmg', 'vaio', 'samsung', 'panasonic ', 'nec ', 'gateway',
          'google', 'fujitsu', 'eurocom', 'asus', 'alienware', 'dell','aoson','gemei','msi',
          'lenovo', 'acer', 'asus', 'hp', 'lg ', 'microsoft','apple']

cpu_brands = ['intel', 'amd']

intel_cores = [' i3', ' i5', ' i7', '2 duo', 'celeron', 'pentium', 'centrino']
amd_cores = ['e-series', 'a8', 'radeon', 'athlon', 'turion', 'phenom']

families = {
    'hp': [r'elitebook', r'compaq', r'folio', r'pavilion'],
    'lenovo': [r' x[0-9]{3}[t]?', r'x1 carbon'],
    'dell': [r'inspiron'],
    'asus': [r'zenbook', ],
    'acer': [r'aspire', r'extensa', ],
    '0': []
}
families_brand = {
    'elitebook': 'hp',
    'compaq': 'hp',
    'folio': 'hp',
    'pavilion': 'hp',
    'inspiron': 'dell',
    'zenbook': 'asus',
    'aspire': 'acer',
    'extensa': 'acer',
    'thinkpad': 'lenovo',
    'thinkcentre': 'lenovo',
    'thinkserver' :'lenovo',
    'toughbook': 'panasonic',
    'envy': 'hp',
    'macbook': 'apple',
    'probook': 'hp',
    'latitude': 'dell',
    'chromebook': '0',
    'tecra':'toshiba',
    'touchsmart':'hp',
    'dominator':'msi',
    'satellite':'toshiba'
}


def clean(data):
    """Clean X2.csv data to a readable format.
    :param data: X2.csv
    :return:
        A DataFrame which contains following columns:
        {instance_id: instance_id of items;
         brand: computer's brand, range in: {'dell', 'lenovo', 'acer', 'asus', 'hp'};
         cpu_brand: cpu's brand, range in: {'intel', 'amd'};
         cpu_core: cpu extra information, relative to cpu_brand;
         cpu_model: cpu model, relative to cpu_brand;
         cpu_frequency: cpu's frequency, unit in Hz;
         ram_capacity: capacity of RAM, unit in GB;
         display_size: size of computer;
         pc_name: name information extract from title;
         name_family: family name of computer;
         title: title information of instance}
         it the value can't extract from the information given, '0' will be filled.
    """

    instance_ids = data.filter(items=['id'], axis=1)
    titles = data.filter(items=['title'], axis=1)
    instance_ids = instance_ids.values.tolist()
    titles = titles.values.tolist()

    result = []
    for row in range(len(instance_ids)):
        # title of each row
        if len(titles[row]) == 0:
            titles[row] = ['']
        rowinfo = titles[row][0]

        brand = '0'
        cpu_brand = '0'
        cpu_core = '0'
        cpu_model = '0'
        cpu_frequency = '0'
        ram_capacity = '0'
        display_size = '0'
        name_number = '0'
        name_family = '0'
        brand_list = []

        # lower_item = ' '.join(sorted(rowinfo.lower().split()))
        lower_item = rowinfo.lower()
        name_info = rowinfo

        for b in brands:
            if b in lower_item:
                brand = b
                brand_list.append(b)

        for family in families_brand.keys():
            if family in lower_item and brand == '0':
                brand = families_brand[family]
                name_family = family
                brand_list.append(brand)
                break
        # 错别字
        if 'pansonic' in lower_item:
            brand = 'panasonic'


        for b in cpu_brands:
            if b in lower_item:
                cpu_brand = b
                break
        if cpu_brand != 'intel':
            for b in amd_cores:
                if b in lower_item:
                    cpu_core = b.strip()
                    # 补充信息
                    cpu_brand = 'amd'
                    break
        if cpu_brand != 'amd':
            for b in intel_cores:
                if b in lower_item:
                    cpu_core = b.strip()
                    cpu_brand = 'intel'
                    break

        if brand == 'lenovo':
            # print(name_info)
            result_name_number = re.search(
                r'[\- ][0-9]{4}[0-9a-zA-Z]{2}[0-9a-yA-Y](?![0-9a-zA-Z])', name_info)
            if result_name_number is None:
                result_name_number = re.search(
                    r'[\- ][0-9]{4}(?![0-9a-zA-Z])', name_info)
                # print(result_name_number.group())
            if result_name_number is not None:
                # print(name_info)
                # print(result_name_number.group())
                name_number = result_name_number.group().replace(
                    '-', '').strip().lower()[:4]
        elif brand == 'hp':
            # print(name_info)
            result_name_number = re.search(r'[0-9]{4}[pPwW]', name_info)
            if result_name_number is None:
                result_name_number = re.search(
                    r'15[\- ][a-zA-Z][0-9]{3}[a-zA-Z]{2}', name_info)
            if result_name_number is None:
                result_name_number = re.search(r'[\s]810[\s](G2)?', name_info)
            if result_name_number is None:
                result_name_number = re.search(r'[0-9]{4}[mM]', name_info)
            if result_name_number is None:
                result_name_number = re.search(
                    r'((DV)|(NC))[0-9]{4}', name_info)
            if result_name_number is None:
                result_name_number = re.search(r'[0-9]{4}DX', name_info)
            if result_name_number is not None:
                # print(result_name_number.group())
                name_number = result_name_number.group().lower().replace('-', '').replace(' ', '')
        elif brand == 'dell':
            # print(name_info)
            result_name_number = re.search(
                r'[a-zA-Z][0-9]{3}[a-zA-Z]', name_info)
            if result_name_number is None:
                result_name_number = re.search(r'[0-9]{3}-[0-9]{3}', name_info)
            if result_name_number is not None:
                # print(result_name_number.group())
                name_number = result_name_number.group().lower().replace('-', '')
        elif brand == 'acer':
            # print(name_info)
            result_name_number = re.search(
                r'[A-Za-z][0-9][\- ][0-9]{3}', name_info)
            if result_name_number is None:
                result_name_number = re.search(r'AS[0-9]{4}', name_info)
            if result_name_number is None:
                result_name_number = re.search(
                    r'[0-9]{4}[- ][0-9]{4}', name_info)
            if result_name_number is not None:
                # print(result_name_number.group())
                name_number = result_name_number.group().lower().replace(' ', '-').replace('-', '')
                if len(name_number) == 8:
                    # print(name_number)
                    name_number = name_number[:4]
        elif brand == 'asus':
            # print(name_info)
            result_name_number = re.search(
                r'[A-Za-z]{2}[0-9]?[0-9]{2}[A-Za-z]?[A-Za-z]', name_info)
            if result_name_number is not None:
                # print(result_name_number.group())
                name_number = result_name_number.group().lower().replace(' ', '-').replace('-', '')

        if cpu_brand == 'intel':
            item_curr = name_info.replace(
                name_number, '').replace(
                name_number.upper(), '')
            # print(item_curr)
            result_model = re.search(
                r'[\- ][0-9]{4}[Qq]?[MmUu](?![Hh][Zz])', item_curr)
            if result_model is None:
                result_model = re.search('[\\- ][0-9]{3}[Qq]?[Mm]', item_curr)
            if result_model is None:
                result_model = re.search('[\\- ][MmQq][0-9]{3}', item_curr)
            if result_model is None:
                result_model = re.search('[\\- ][PpNnTt][0-9]{4}', item_curr)
            if result_model is None:
                result_model = re.search('[\\- ][0-9]{4}[Yy]', item_curr)
            if result_model is None:
                result_model = re.search('[\\- ][Ss]?[Ll][0-9]{4}', item_curr)
            if result_model is None:
                result_model = re.search('[\\- ]867', item_curr)
            if result_model is None:
                result_model = re.search(
                    '[\\- ]((1st)|(2nd)|(3rd)|([4-9]st))[ ][Gg]en', item_curr)
            if result_model is not None:
                cpu_model = result_model.group()[1:].lower()
                # print(cpu_model)
        elif cpu_brand == 'amd':
            item_curr = name_info.replace(
                name_number, '').replace(
                name_number.upper(), '')
            if cpu_core == 'a8':
                cpu_core = 'a-series'
            result_model = re.search(r'([AaEe][0-9][\- ][0-9]{4})', item_curr)
            if result_model is None:
                result_model = re.search('[\\- ]HD[\\- ][0-9]{4}', item_curr)
            if result_model is None:
                result_model = re.search(
                    '[\\- ][AaEe][\\- ][0-9]{3}', item_curr)
            if result_model is not None:
                cpu_core = result_model.group().replace(
                    '-', '').replace(' ', '')[:1].lower() + '-series'
                cpu_model = result_model.group()[1:].lower().replace(' ', '-')
            if cpu_core in ('radeon', 'athlon', 'turion', 'phenom'):
                if result_model is None:
                    result_model = re.search('[\\- ][NnPp][0-9]{3}', item_curr)
                if result_model is None:
                    result_model = re.search(
                        '[\\- ](64[ ]?[Xx]2)|([Nn][Ee][Oo])', item_curr)
                if result_model is not None:
                    cpu_model = result_model.group().lower().replace('-', '').replace(' ', '')

        result_frequency = re.search(
            r'[123][ .][0-9]?[0-9]?[ ]?[Gg][Hh][Zz]', name_info)
        if result_frequency is not None:
            # print(result_frequency.group())
            result_frequency = re.split(r'[GgHhZz]', result_frequency.group())[
                0].strip().replace(' ', '.')
            if len(result_frequency) == 3:
                result_frequency = result_frequency + '0'
            if len(result_frequency) == 1:
                result_frequency = result_frequency + '.00'
            result_frequency = result_frequency
            cpu_frequency = result_frequency

        result_ram_capacity = re.search(
            r'[1-9][\s]?[Gg][Bb][\s]?((S[Dd][Rr][Aa][Mm])|(D[Dd][Rr]3)|([Rr][Aa][Mm])|(Memory))', name_info)
        if result_ram_capacity is not None:
            # print(result_ram_capacity.group())
            ram_capacity = result_ram_capacity.group()[:1]

        result_display_size = re.search(r'1[0-9]([. ][0-9])?\"', name_info)
        if result_display_size is not None:
            display_size = result_display_size.group().replace(" ", ".")[:-1]
        else:
            result_display_size = re.search(
                r'1[0-9]([. ][0-9])?[- ][Ii]nch(?!es)', name_info)
        if result_display_size is not None and display_size == '0':
            display_size = result_display_size.group().replace(" ", ".")[:-5]
        elif result_display_size is None:
            result_display_size = re.search(
                r'(?<!x)[ ]1[0-9][. ][0-9]([ ]|(\'\'))(?!x)', name_info)
        if result_display_size is not None and display_size == '0':
            display_size = result_display_size.group().replace(
                "\'", " ").strip().replace(' ', '.')

        if brand in families.keys():
            for pattern in families[brand]:
                result_name_family = re.search(pattern, lower_item)
                if result_name_family is not None:
                    name_family = result_name_family.group().strip()
                    break

        result.append([
            instance_ids[row][0],
            brand,
            cpu_brand,
            cpu_core,
            cpu_model,
            cpu_frequency,
            ram_capacity,
            display_size,
            name_number,
            name_family,
            titles[row][0].lower()
        ])
        # 标注
        names = ['brand', 'cpu_brand', 'cpu_core', 'cpu_model', 'cpu_frequency', 'ram_capacity', 'display_size',
                 'name_number', 'name_family']
        tag = []
        mapping = {'brand': brand,
                   'cpu_brand': cpu_brand,
                   'cpu_core': cpu_core,
                   'cpu_model': cpu_model,
                   'cpu_frequency': cpu_frequency,
                   'ram_capacity': ram_capacity,
                   'display_size': display_size,
                   'name_number': name_number,
                   'name_family': name_family}
        words = titles[row][0].lower().split()
        for i in range(len(words)):
            tag.append('0')
        for name in names:
            if name == 'brand':
                for x in brand_list:
                    if x.strip() in words:
                        index = words.index(x.strip())
                        tag[index] = 'B-' + 'brand'
            else:
                if mapping[name] != '0':
                    if mapping[name] in words:
                        index = words.index(mapping[name])
                        tag[index] = 'B-' + name

        # print(tag)

    result = pd.DataFrame(result)
    name = [
        'instance_id',
        'brand',
        'cpu_brand',
        'cpu_core',
        'cpu_model',
        'cpu_frequency',
        'ram_capacity',
        'display_size',
        'pc_name',
        'family',
        'title'
    ]
    for i in range(len(name)):
        result.rename({i: name[i]}, inplace=True, axis=1)

    return result
