import pandas as pd
import re


def extract_x1(data):
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

         if the value can't extract from the information given, '0' will be filled.
    """
    brands = ['dell', 'lenovo', 'acer', 'asus', 'hp']

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

    instance_ids = data.filter(items=['id'], axis=1)
    titles = data.filter(items=['title'], axis=1)
    information = data.drop(['id'], axis=1)
    information = information.fillna('')
    instance_ids = instance_ids.values.tolist()
    information = information.values.tolist()
    titles = titles.values.tolist()

    result = []
    for row in range(len(instance_ids)):
        information[row].sort(key=lambda i: len(i), reverse=True)
        rowinfo = titles[row][0]
        for mess in information[row]:
            if mess not in rowinfo:
                rowinfo = rowinfo + ' - ' + mess

        brand = '0'
        cpu_brand = '0'
        cpu_core = '0'
        cpu_model = '0'
        cpu_frequency = '0'
        ram_capacity = '0'
        display_size = '0'
        name_number = '0'
        name_family = '0'

        item = rowinfo
        lower_item = item.lower()

        name_info = item

        for b in brands:
            if b in lower_item:
                brand = b
                break

        for b in cpu_brands:
            if b in lower_item:
                cpu_brand = b
                break
        if cpu_brand != 'intel':
            for b in amd_cores:
                if b in lower_item:
                    cpu_core = b.strip()
                    cpu_brand = 'amd'
                    break
        if cpu_brand != 'amd':
            for b in intel_cores:
                if b in lower_item:
                    cpu_core = b.strip()
                    cpu_brand = 'intel'
                    break

        if brand == 'lenovo':
            result_name_number = re.search(
                r'[\- ][0-9]{4}[0-9a-zA-Z]{2}[0-9a-yA-Y](?![0-9a-zA-Z])', name_info)
            if result_name_number is None:
                result_name_number = re.search(
                    r'[\- ][0-9]{4}(?![0-9a-zA-Z])', name_info)
            if result_name_number is not None:
                name_number = result_name_number.group().replace(
                    '-', '').strip().lower()[:4]
        elif brand == 'hp':
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
                name_number = result_name_number.group().lower().replace('-', '').replace(' ', '')
        elif brand == 'dell':
            result_name_number = re.search(
                r'[a-zA-Z][0-9]{3}[a-zA-Z]', name_info)
            if result_name_number is None:
                result_name_number = re.search(r'[0-9]{3}-[0-9]{3}', name_info)
            if result_name_number is not None:
                name_number = result_name_number.group().lower().replace('-', '')
        elif brand == 'acer':
            result_name_number = re.search(
                r'[A-Za-z][0-9][\- ][0-9]{3}', name_info)
            if result_name_number is None:
                result_name_number = re.search(r'AS[0-9]{4}', name_info)
            if result_name_number is None:
                result_name_number = re.search(
                    r'[0-9]{4}[- ][0-9]{4}', name_info)
            if result_name_number is not None:
                name_number = result_name_number.group().lower().replace(' ', '-').replace('-', '')
                if len(name_number) == 8:
                    name_number = name_number[:4]
        elif brand == 'asus':
            result_name_number = re.search(
                r'[A-Za-z]{2}[0-9]?[0-9]{2}[A-Za-z]?[A-Za-z]', name_info)
            if result_name_number is not None:
                name_number = result_name_number.group().lower().replace(' ', '-').replace('-', '')

        if cpu_brand == 'intel':
            item_curr = item.replace(
                name_number, '').replace(
                name_number.upper(), '')
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
        elif cpu_brand == 'amd':
            item_curr = item.replace(
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
            r'[123][ .][0-9]?[0-9]?[ ]?[Gg][Hh][Zz]', item)
        if result_frequency is not None:
            result_frequency = re.split(r'[GgHhZz]', result_frequency.group())[
                0].strip().replace(' ', '.')
            if len(result_frequency) == 3:
                result_frequency = result_frequency + '0'
            if len(result_frequency) == 1:
                result_frequency = result_frequency + '.00'
            result_frequency = result_frequency
            cpu_frequency = result_frequency

        result_ram_capacity = re.search(
            r'[1-9][\s]?[Gg][Bb][\s]?((S[Dd][Rr][Aa][Mm])|(D[Dd][Rr]3)|([Rr][Aa][Mm])|(Memory))', item)
        if result_ram_capacity is not None:
            ram_capacity = result_ram_capacity.group()[:1]

        result_display_size = re.search(r'1[0-9]([. ][0-9])?\"', item)
        if result_display_size is not None:
            display_size = result_display_size.group().replace(" ", ".")[:-1]
        else:
            result_display_size = re.search(
                r'1[0-9]([. ][0-9])?[- ][Ii]nch(?!es)', item)
        if result_display_size is not None and display_size == '0':
            display_size = result_display_size.group().replace(" ", ".")[:-5]
        elif result_display_size is None:
            result_display_size = re.search(
                r'(?<!x)[ ]1[0-9][. ][0-9]([ ]|(\'\'))(?!x)', item)
        if result_display_size is not None and display_size == '0':
            display_size = result_display_size.group().replace(
                "\'", " ").strip().replace(' ', '.')

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

    result = pd.DataFrame(result)
    name = [
        'id',
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


def extract_x2(data):
    """Clean X4.csv data to a readable format.

    :param data: X4.csv

    :return:
        A DataFrame which contains following columns:
        {instance_id: instance_id of items;
         brand: item's brand, for example: {'intenso', 'pny', 'lexar'}
         capacity: usb/sd card's capacity, unit in GB
         price: price of the item
         mem_type: memory type, for example: {'ssd', 'sd', 'microsd', 'usb'}
         type: type information, relative to brand
         model: model information, relative to brand
         item_code: the unique item code
         title: title information of instance}

         it the value can't extract from the information given, '0' will be filled.
    """
    brand_list = [
        "intenso",
        "pny",
        "lexar",
        "sony",
        "sandisk",
        "kingston",
        "samsung",
        "toshiba",
        "transcend"]

    intenso_type = ["basic", "rainbow", "high speed", "speed", "premium", "alu", "business", "micro",
                    "imobile", "cmobile", "mini", "ultra", "slim", "flash", "mobile"]

    colors = ['midnight black', 'prism white', 'prism black', 'prism green', 'prism blue', 'canary yellow',
              'flamingo pink', 'cardinal red', 'smoke blue', 'deep blue', 'coral orange',
              'black sky', 'gold sand', 'blue mist and peach cloud', 'orchid gray',
              'metallic copper', 'lavender purple', 'ocean blue', 'pure white', 'alpine white',
              'copper', 'red', 'black', 'blue', 'white', 'silver', 'gold', 'violet', 'purple',
              'brown', 'orange', 'coral', 'pink']

    names = data.filter(items=['name'], axis=1).fillna('')
    prices = data.filter(items=['price'], axis=1).fillna('')
    brands = data.filter(items=['brand'], axis=1).fillna('')
    # descriptions = data.filter(items=['description'], axis=1).fillna('')
    # categories = data.filter(item=['description'], axis=1).fillna('')
    instance_ids = data.filter(items=['id'], axis=1)
    names = names.values.tolist()
    prices = prices.values.tolist()
    brands = brands.values.tolist()
    instance_ids = instance_ids.values.tolist()

    result = []

    for row in range(len(instance_ids)):
        nameinfo = names[row][0].lower()

        size = '0'
        price = '0'
        mem_type = '0'
        brand = '0'
        type = '0'
        model = '0'
        item_code = '0'

        size_model = re.search(r'[0-9]{1,4}[ ]*[gt][bo]', nameinfo)
        if size_model is not None:
            size = size_model.group()[:].lower().replace(' ', '')

        if prices[row][0] != '':
            price = prices[row][0]

        if brands[row][0] != '':
            brand = brands[row][0].lower()
        else:
            for b in brands:
                if b[0] in nameinfo:
                    brand = b[0]
                    break

        mem_model = re.search(r'ssd', nameinfo)
        if mem_model is None:
            mem_model = re.search(r'micro[- ]?sd[hx]?c?', nameinfo)
        if mem_model is None:
            mem_model = re.search(r'sd[hx]c', nameinfo)
        if mem_model is None:
            mem_model = re.search(r'usb', nameinfo)
        if mem_model is None:
            mem_model = re.search(r'sd(?!cz)', nameinfo)
        if mem_model is None:
            mem_model = re.search(r'secure digital', nameinfo)
        if mem_model is None:
            mem_model = re.search(r'xqd', nameinfo)
        if mem_model is None:
            mem_model = re.search(r'ljd', nameinfo)
        if mem_model is not None:
            mem_type = mem_model.group()
            if mem_type not in ('ssd', 'usb', 'xqd'):
                if 'micro' in mem_type:
                    mem_type = 'microsd'
                elif 'ljd' in mem_type:
                    mem_type = 'usb'
                else:
                    mem_type = 'sd'
        if ('adapter' in nameinfo or 'adaptateur' in nameinfo or 'adaptador' in nameinfo) and mem_type == '0':
            mem_type = 'microsd'

        item_code_model = re.search(r'\((mk)?[0-9]{6,10}\)', nameinfo)
        if item_code_model is not None:
            item_code = item_code_model.group()[1:-1]

        if brand == "intenso":
            model_model = re.search(r'[0-9]{7}', nameinfo)
            if model_model is not None:
                model = model_model.group()[:]

            type_model = re.search(r'(high\s)?[a-z]+\s(?=line)', nameinfo)
            if type_model is not None:
                type = type_model.group()[:].replace(' ', '')
            else:
                for t in intenso_type:
                    if t in nameinfo:
                        type = t.replace(' ', '')
                        break

        elif brand == "lexar":
            type_model = re.search(
                r'((jd)|[\s])[a-wy-z][0-9]{2}[a-z]?', nameinfo)
            if type_model is None:
                type_model = re.search(r'[\s][0-9]+x(?![a-z0-9])', nameinfo)
            if type_model is None:
                type_model = re.search(r'(([\s][x])|(beu))[0-9]+', nameinfo)
            if type_model is not None:
                type = type_model.group().strip() \
                    .replace('x', '').replace('l', '').replace('j', '').replace('d', '') \
                    .replace('b', '').replace('e', '').replace('u', '')

            if mem_type == '0':
                if 'drive' in nameinfo:
                    mem_type = 'usb'
            if 'lexar 8gb jumpdrive v10 8gb usb 2.0 tipo-a blu unit?? flash usb' in nameinfo:
                type = 'c20c'

        elif brand == 'sony':
            if mem_type == '0':
                if 'ux' in nameinfo or 'uy' in nameinfo or 'sr' in nameinfo:
                    mem_type = 'microsd'
                elif 'uf' in nameinfo:
                    mem_type = 'sd'
                elif 'usm' in nameinfo or size == '1tb':
                    mem_type = 'usb'

            type_model = re.search(r'((sf)|(usm))[-]?[0-9a-z]{1,6}', nameinfo)
            if type_model is not None:
                type = type_model.group().replace('-', '').replace('g', '')
                for c in range(ord('0'), ord('9')):
                    type = type.replace(chr(c), '')
                if 'sf' in type and mem_type == '0':
                    mem_type = 'sd'
                type = type.replace('sf', '').replace('usm', '')
            elif mem_type in ('sd', 'usb'):
                if 'machqx' in nameinfo:
                    type = 'qx'
                elif 'type-c' in nameinfo or 'type c' in nameinfo:
                    type = 'ca'
                type_model = re.search(
                    r'(serie[s]?[\s-]?[a-z]{1,2}[\s])|([\s][a-z]{1,2}[\-]?serie[s]?)', nameinfo)
                if type_model is not None:
                    type = type_model.group().replace(
                        ' ',
                        '').replace(
                        '-',
                        '').replace(
                        'g',
                        '')
                    type = type.replace('series', '').replace('serie', '')

        elif brand == 'sandisk':
            model_model = re.search(r'ext.*(\s)?((plus)|(pro)|\+)', nameinfo)
            if model_model is not None:
                model = 'ext+'
            else:
                model_model = re.search(r'ext(reme)?', nameinfo)
                if model_model is not None:
                    model = 'ext'
                else:
                    model_model = re.search(r'fit', nameinfo)
                    if model_model is None:
                        model_model = re.search(r'glide', nameinfo)
                    if model_model is None:
                        model_model = re.search(r'blade', nameinfo)
                    if model_model is not None:
                        model = model_model.group()
                    else:
                        model_model = re.search(
                            r'ultra(\s)?((plus)|(pro)|\+|(performance)|(android))', nameinfo)
                        if model_model is None:
                            model_model = re.search(
                                r'sandisk 8gb ultra sdhc memory card, class 10, read speed up to 80 mb/s \+ sd adapter',
                                nameinfo)
                        if model_model is None:
                            model_model = re.search(
                                r'sandisk sdhc [0-9]+gb 80mb/s cl10\\n', nameinfo)
                        if model_model is not None:
                            model = 'ultra+'
                        else:
                            model_model = re.search(r'ultra', nameinfo)
                            if model_model is not None:
                                model = 'ultra'
                            else:
                                model_model = re.search(r'dual', nameinfo)
                                if model_model is None:
                                    model_model = re.search(
                                        r'double connect.*', nameinfo)
                                if model_model is not None:
                                    model = 'ultra'

            if 'accessoires montres' in nameinfo:
                if 'extreme' in nameinfo:
                    mem_type = 'microsd'
                    model = 'ultra+'
                elif 'ext pro' in nameinfo:
                    mem_type = 'microsd'
                    model = 'ext+'
            if 'adapter' in nameinfo or 'adaptateur' in nameinfo:
                mem_type = 'microsd'
            if mem_type == '0':
                if 'drive' in nameinfo:
                    mem_type = 'usb'
                elif 'cruzer' in nameinfo:
                    mem_type = 'usb'
                elif model in ('glide', 'fit'):
                    mem_type = 'usb'
            if 'sandisk - ' + size + ' extreme en fnac.es' in nameinfo:
                mem_type = 'usb'

        elif brand == 'pny':
            type_model = re.search(r'att.*?[3-4]', nameinfo)
            if type_model is not None:
                type = type_model.group().replace(' ', '').replace('-', '')
                type = 'att' + \
                       list(filter(lambda ch: ch in '0123456789', type))[0]
                if mem_type == '0':
                    mem_type = 'usb'

        elif brand == 'kingston':
            if mem_type == '0':
                if 'savage' in nameinfo or 'hx' in nameinfo or 'hyperx' in nameinfo:
                    mem_type = 'usb'
                elif 'ultimate' in nameinfo:
                    mem_type = 'sd'
            model_model = re.search(
                r'(dt[i1]0?1?)|(data[ ]?t?travel?ler)', nameinfo)
            if model_model is not None:
                model = 'data traveler'
                type_model = re.search(r'(g[234])|(gen[ ]?[234])', nameinfo)
                if type_model is not None:
                    type = type_model.group()[-1:]
            else:
                type_model = re.search(
                    r'[\s]((g[234])|(gen[ ]?[234]))[\s]', nameinfo)
                if type_model is not None:
                    type = type_model.group().strip()[-1:]
                    model = 'data traveler'
            if model == 'data traveler' and mem_type == '0':
                mem_type = 'usb'

        elif brand == 'samsung':
            if 'lte' in nameinfo:
                model_model = re.search(
                    r'[\s][a-z][0-9]{1,2}[a-z]?[\s]((plus)|\+)?', nameinfo)
                if model_model is None:
                    model_model = re.search(
                        r'[\s]note[\s]?[0-9]{1,2}\+?[\s]?(ultra)?', nameinfo)
                if model_model is None:
                    model_model = re.search(r'prime[ ]?((plus)|\+)?', nameinfo)
                if model_model is not None:
                    model = model_model.group().replace(' ', '').replace('plus', '+')
                mem_type = 'sim'
            elif 'tv' in nameinfo:
                size_model = re.search(r'[0-9]{2}[- ]?inch', nameinfo)
                if size_model is not None:
                    size = size_model.group()[:2]
                mem_model = re.search(r'(hd)|(qled)|(uhd)', nameinfo)
                if mem_model is not None:
                    mem_type = mem_model.group()
                model_model = re.search(r'[a-z]{1,2}[0-9]{4}', nameinfo)
                if model_model is not None:
                    model = model_model.group()
            else:
                if mem_type == 'ssd':
                    model_model = re.search(r'[\s]t[0-9][\s]', nameinfo)
                    if model_model is not None:
                        model = model_model.group().strip()
                else:
                    model_model = re.search(r'(pro)|(evo)', nameinfo)
                    if model_model is not None:
                        model = model_model.group()
                        model_model = re.search(r'(\+)|(plus)', nameinfo)
                        if model_model is not None:
                            model = model + model_model.group().replace('plus', '+')
                        if model == 'evo+' and mem_type == '0':
                            mem_type = 'microsd'
            for c in colors:
                if c in nameinfo:
                    type = c
                    break

        elif brand == 'toshiba':
            model_model = re.search(r'[\s\-n][umn][0-9]{3}', nameinfo)
            if model_model is not None:
                model = model_model.group()[1:]
                if mem_type == '0':
                    ch = model[0]
                    if ch == 'u':
                        mem_type = 'usb'
                    elif ch == 'n':
                        mem_type = 'sd'
                    elif ch == 'm':
                        mem_type = 'microsd'
            if mem_type == 'usb' and model == '0':
                model_model = re.search(r'ex[\s-]?ii', nameinfo)
                if model_model is None:
                    model_model = re.search(r'osus', nameinfo)
                if model_model is not None:
                    model = 'ex'
            if 'transmemory' in nameinfo:
                if mem_type == '0':
                    mem_type = 'usb'
            if mem_type != 'usb':
                type_model = re.search(
                    r'exceria[ ]?((high)|(plus)|(pro))?', nameinfo)
                if type_model is not None:
                    type = type_model.group().replace(' ', '').replace('exceria', 'x')
                elif size != '0':
                    type_model = re.search(
                        r'x[ ]?((high)|(plus)|(pro))?' + size[:-2], nameinfo)
                    if type_model is not None:
                        type = type_model.group().replace(' ', '')[
                               :-(len(size) - 2)]
                if type == 'xpro' and mem_type == '0':
                    mem_type = 'sd'
                if type == 'xhigh' and mem_type == '0':
                    mem_type = 'microsd'
            if mem_type == 'usb' and model == '0':
                if 'hayaqa' in nameinfo or 'hayabusa' in nameinfo:
                    model = 'u202'
            if mem_type == 'sd' and model == '0':
                model_model = re.search(r'silber', nameinfo)
                if model_model is not None:
                    model = 'n401'
            if mem_type == 'sd' and model == '0' and type == '0':
                model_model = re.search(
                    r'sd[hx]c uhs clas[se] 3 memor(y|(ia)) ((card)|(flash))',
                    nameinfo)
                if model_model is not None:
                    type = 'xpro'
            if mem_type == 'sd' and model == '0' and type == '0':
                if 'uhs-ii' in nameinfo and 'carte m??moire flash' in nameinfo:
                    type = 'xpro'
            if mem_type != 'usb':
                speed_model = re.search(
                    r'[1-9][0-9]{1,2}[\s]?m[bo]/s', nameinfo)
                if speed_model is not None:
                    speed = re.search(
                        r'[0-9]{2,3}', speed_model.group()).group()
                    if speed == '260' or speed == '270':
                        if type == '0':
                            type = 'xpro'
                    if speed == '90' and type == 'x':
                        if model == '0':
                            model = 'n302'
            if 'gb uhs-i (u3 - up to 95mb/s read) flash memory card' in nameinfo:
                mem_type = 'microsd'
                type = 'x'
            if 'toshiba pendrive usb high-speed' in nameinfo:
                model = 'u202'
            if 'en fnac.es' in nameinfo and 'toshiba usb 3.0' in nameinfo and 'pendrive / memoria usb' in nameinfo:
                model = 'ex'
            if model == 'n101':
                model = '0'

        elif brand == 'transcend':
            pass

        result.append([
            instance_ids[row][0],
            brand,
            size,
            price,
            mem_type,
            type,
            model,
            item_code,
            nameinfo
        ])

    result = pd.DataFrame(result)

    name = [
        'instance_id',
        'brand',
        'capacity',
        'price',
        'mem_type',
        'type',
        'model',
        'item_code',
        'title']
    for i in range(len(name)):
        result.rename({i: name[i]}, inplace=True, axis=1)

    return result
