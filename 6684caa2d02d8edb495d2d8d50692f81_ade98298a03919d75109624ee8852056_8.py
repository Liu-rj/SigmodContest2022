import pandas as pd
import re


brands = ['sandisk',
          'sony',
          'lexar',
          'samsung',
          'toshiba',
          'intenso',
          'kingston',
          'accessoires',
          'pny',
          'carte',
          'transcend',
          'karta',
          'dorr',
          'wysokiej',
          'tarjeta',
          'sdisk',
          'achat',
          'tesco',
          'chiavetta',
          'clé']

uncommon_brands = ['clé', 'sdsdunc-064g-gzfin', 'memoria ultra', 'sddd3-128g-g46', 'lsdmi16gbbeu633a', 'premium bracelets']

intenso_type = ['3521491', 'ultra memoria memory', 'extreme', 'cs/ultra', 'basic', 'rainbow', 'high speed', 'speed',
                'premium', 'alu',
                'business', 'micro', 'tarjeta uhs1', 'carte memoria classe', 'achat tarjeta flashgeheugenkaart',
                'imobile', 'cmobile', 'mini', 'ultra', 'slim', 'flash', 'mobile', 'secure', 'transmemory',
                'prime', 'polecana']

colors = ['midnight black', 'prism white', 'prism black', 'prism green', 'prism blue', 'canary yellow',
          'flamingo pink', 'cardinal red', 'smoke blue', 'deep blue', 'coral orange',
          'black sky', 'gold sand', 'blue mist and peach cloud', 'orchid gray',
          'metallic copper', 'lavender purple', 'ocean blue', 'pure white', 'alpine white',
          'copper', 'red', 'black', 'blue', 'white', 'silver', 'gold', 'violet', 'purple',
          'brown', 'orange', 'coral', 'pink']

instances = ['disque dur', 'hdsl1', 'hdsl-1', 'hd-sl', 'usb-flash-laufwerk',
             'usb-minne', 'usb 3', 'usb 2', 'usb3', 'usb2', 'usb', 'micro-sdhc',
             'sd', 'memoria de', 'dual sim', 'usm32gr', 'usm', 'xqd', 'uhs-i',
             'uhs 1', 'uhs-1', 'uhs-ii', 'memóriakártya card', 'transmemory',
             'exceria pro', 'flashdisk', 'flashgeheugenkaart', 'microduo xc',
             'card', 'speicherkarte', 'micro x', 'karty', 'hasta', 'memoria flash',
             'micro 32 drive 10', 'adattatore lecture rosa', 'cruzer fit']


def is_power(n):
    res = 1
    while res < n:
        res = res << 1
    if res == n:
        return True
    else:
        return False


def clean_x2(data):
    id_list = data['id'].tolist()
    name_list = data['name'].fillna('').map(lambda x: str(x).lower()).tolist()
    price_list = data['price'].fillna(0).map(lambda x: str(x).lower()).tolist()
    brand_list = data['brand'].fillna('').map(lambda x: str(x).lower()).tolist()
    description_list = data['description'].fillna('').map(lambda x: str(x).lower()).tolist()

    result = []
    for row in range(len(id_list)):
        brand = brand_list[row]
        name = name_list[row].replace('\\n', '')
        price = price_list[row]
        description = description_list[row]

        ''' Extracting the brand '''
        for brand_name in brands:
            if brand_name in name:
                brand = brand_name
                break
        if not brand:
            for brand_name in uncommon_brands:
                if brand_name in name:
                    brand = brand_name
                    break
        if not brand and description:
            for brand_name in brands:
                if brand_name in description:
                    brand = brand_name
                    break
        if not brand and description:
            for brand_name in uncommon_brands:
                if brand_name in description:
                    brand = brand_name
                    break
        if not brand:
            if 'hyperx' in name + description and 'savage' in name + description:
                brand = 'kingston'
        if brand and brand not in name:
            name += ' ' + brand

        ''' Extracting storage capacity '''
        capacity = ''
        if re.search(r'\d+[ ]*[gt][bo]', name):
            capacity = re.search(r'\d+[ ]*[gt][bo]', name).group().replace(' ', '').replace('o', 'b')
        elif re.search(r'([gt][bo])([ ]*)(\d+)', name):
            temp = re.search(r'([gt][bo])([ ]*)(\d+)', name)
            if is_power(int(temp.group(3))):
                capacity = temp.group().replace(' ', '').replace('o', 'b').replace('gb', '') + 'gb'
        elif re.search(r'\d+[ ]*[gt][bo]', description_list[row]):
            capacity = re.search(r'\d+[ ]*[gt][bo]', description_list[row]).group().replace(' ', '').replace('o', 'b')
        elif re.search(r'([gt][bo])([ ]*)(\d+)', description_list[row]):
            temp = re.search(r'([gt][bo])([ ]*)(\d+)', description_list[row])
            if is_power(int(temp.group(3))):
                capacity = temp.group().replace(' ', '').replace('o', 'b').replace('gb', '') + 'gb'

        ''' Extracting instance type '''
        instance_type = ''
        instance_pattern = None
        if re.search(r'ssd', name):
            instance_pattern = re.search(r'ssd', name)
        elif re.search(r'micro[- ]?sd[hx]?c?', name):
            instance_pattern = re.search(r'micro[- ]?sd[hx]?c?', name)
        elif re.search(r'sd[hx]c', name):
            instance_pattern = re.search(r'sd[hx]c', name)
        elif re.search(r'usb', name):
            instance_pattern = re.search(r'usb', name)
        elif re.search(r'sd(?!cz)', name):
            instance_pattern = re.search(r'sd(?!cz)', name)
        elif re.search(r'secure digital', name):
            instance_pattern = re.search(r'secure digital', name)
        elif re.search(r'xqd', name):
            instance_pattern = re.search(r'xqd', name)
        elif re.search(r'ljd', name):
            instance_pattern = re.search(r'ljd', name)

        if instance_pattern:
            instance_type = instance_pattern.group()
            if instance_type in ['hxsav', 'hyperx savage', 'hx savage']:
                instance_type = 'hyperx savage'
            if instance_type not in ['ssd', 'usb', 'xqd', 'hyperx savage']:
                if 'micro' in instance_type:
                    instance_type = 'microsd'
                elif 'ljd' in instance_type:
                    instance_type = 'usb'
                else:
                    instance_type = 'sd'
        if ('adapter' in name or 'adaptateur' in name or 'adaptador' in name) and instance_type == '':
            instance_type = 'microsd'
        if re.search(r'hxsav|hyperx savage|hx savage', name):
            instance_type = 'hyperx savage'

        ''' Extracting special product code '''
        item_code = ''
        if re.search(r'\((mk)?[0-9]{6,10}\)', name):
            # print(brand)
            item_code = re.search(r'\((mk)?[0-9]{6,10}\)', name).group()[1:-1]
        elif re.search(r'(lsd)[0-9A-Za-z]+', name):
            item_code = re.search(r'(lsd)[0-9A-Za-z]+', name).group()

        ''' Extracting special product type '''
        type = ''
        model = ''
        if brand == "intenso":
            if re.search(r'[0-9]{7}', name):
                model = re.search(r'[0-9]{7}', name).group()[:]
            if re.search(r'(high\s)?[a-z]+\s(?=line)', name):
                type = re.search(r'(high\s)?[a-z]+\s(?=line)', name).group().replace(' ', '')
            else:
                if model == '3534490' or model == '3534460':
                    type = 'premium'
                elif model == '3503470':
                    type = 'basic'
                elif model == '3502450':
                    type = 'rainbow'
                elif model == '3530460':
                    type = 'speed'
                else:
                    for t in intenso_type:
                        if t in name:
                            type = t.replace(' ', '')
                            break

        elif brand == "lexar":
            type_pattern = re.search(
                r'((jd)|[\s])[a-wy-z][0-9]{2}[a-z]?', name)
            if type_pattern is None:
                type_pattern = re.search(r'[\s][0-9]+x(?![a-z0-9])', name)
            if type_pattern is None:
                type_pattern = re.search(r'(([\s][x])|(beu))[0-9]+', name)
            if type_pattern is not None:
                type = type_pattern.group().strip() \
                    .replace('x', '').replace('l', '').replace('j', '').replace('d', '') \
                    .replace('b', '').replace('e', '').replace('u', '')

            if instance_type == '':
                if 'drive' in name:
                    instance_type = 'usb'
            if 'lexar 8gb jumpdrive v10 8gb usb 2.0 tipo-a blu unità flash usb' in name:
                type = 'c20c'

        elif brand == 'sony':
            if instance_type == '':
                if 'ux' in name or 'uy' in name or 'sr' in name:
                    instance_type = 'microsd'
                elif 'uf' in name:
                    instance_type = 'sd'
                elif 'usm' in name or capacity == '1tb':
                    instance_type = 'usb'

            type_pattern = re.search(r'((sf)|(usm))[-]?[0-9a-z]{1,6}', name)
            if type_pattern is not None:
                type = type_pattern.group().replace('-', '').replace('g', '')
                for c in range(ord('0'), ord('9')):
                    type = type.replace(chr(c), '')
                if 'sf' in type and instance_type == '':
                    instance_type = 'sd'
                type = type.replace('sf', '').replace('usm', '')
            elif instance_type in ('sd', 'usb'):
                if 'machqx' in name:
                    type = 'qx'
                elif 'type-c' in name or 'type c' in name:
                    type = 'ca'
                type_pattern = re.search(
                    r'(serie[s]?[\s-]?[a-z]{1,2}[\s])|([\s][a-z]{1,2}[\-]?serie[s]?)', name)
                if type_pattern is not None:
                    type = type_pattern.group().replace(' ', '').replace('-', '').replace('g', '')
                    type = type.replace('series', '').replace('serie', '')

        elif brand == 'sandisk':
            model_pattern = re.search(r'ext.*(\s)?((plus)|(pro)|\+)', name)
            if model_pattern is not None:
                model = 'ext+'
            else:
                model_pattern = re.search(r'ext(reme)?', name)
                if model_pattern is not None:
                    model = 'ext'
                else:
                    model_pattern = re.search(r'fit', name)
                    if model_pattern is None:
                        model_pattern = re.search(r'glide', name)
                    if model_pattern is None:
                        model_pattern = re.search(r'blade', name)
                    if model_pattern is not None:
                        model = model_pattern.group()
                    else:
                        model_pattern = re.search(
                            r'ultra(\s)?((plus)|(pro)|\+|(performance)|(android))', name)
                        if model_pattern is None:
                            model_pattern = re.search(
                                r'sandisk 8gb ultra sdhc memory card, class 10, read speed up to 80 mb/s \+ sd adapter',
                                name)
                        if model_pattern is None:
                            model_pattern = re.search(
                                r'sandisk sdhc [0-9]+gb 80mb/s cl10\\n', name)
                        if model_pattern is not None:
                            model = 'ultra+'
                        else:
                            model_pattern = re.search(r'ultra', name)
                            if model_pattern is not None:
                                model = 'ultra'
                            else:
                                model_pattern = re.search(r'dual', name)
                                if model_pattern is None:
                                    model_pattern = re.search(
                                        r'double connect.*', name)
                                if model_pattern is not None:
                                    model = 'ultra'

            if 'accessoires montres' in name:
                if 'extreme' in name:
                    instance_type = 'microsd'
                    model = 'ultra+'
                elif 'ext pro' in name:
                    instance_type = 'microsd'
                    model = 'ext+'
            if 'adapter' in name or 'adaptateur' in name:
                instance_type = 'microsd'
            if instance_type == '':
                if 'drive' in name:
                    instance_type = 'usb'
                elif 'cruzer' in name:
                    instance_type = 'usb'
                elif model in ('glide', 'fit'):
                    instance_type = 'usb'
            if 'sandisk - ' + capacity + ' extreme en fnac.es' in name:
                instance_type = 'usb'

        elif brand == 'pny':
            type_pattern = re.search(r'att.*?[3-4]', name)
            if type_pattern is not None:
                type = type_pattern.group().replace(' ', '').replace('-', '')
                type = 'att' + \
                       list(filter(lambda x: x in '0123456789', type))[0]
                if instance_type == '':
                    instance_type = 'usb'

        elif brand == 'kingston':
            if instance_type == '':
                if 'savage' in name or 'hx' in name or 'hyperx' in name:
                    instance_type = 'usb'
                elif 'ultimate' in name:
                    instance_type = 'sd'
            model_pattern = re.search(
                r'(dt[i1]0?1?)|(data[ ]?t?travel?ler)', name)
            if model_pattern is not None:
                model = 'data traveler'
                type_pattern = re.search(r'(g[234])|(gen[ ]?[234])', name)
                if type_pattern is not None:
                    type = type_pattern.group()[-1:]
            else:
                type_pattern = re.search(
                    r'[\s]((g[234])|(gen[ ]?[234]))[\s]', name)
                if type_pattern is not None:
                    type = type_pattern.group().strip()[-1:]
                    model = 'data traveler'
            if model == 'data traveler' and instance_type == '':
                instance_type = 'usb'

        elif brand == 'samsung':
            if 'lte' in name:
                model_pattern = re.search(
                    r'[\s][a-z][0-9]{1,2}[a-z]?[\s]((plus)|\+)?', name)
                if model_pattern is None:
                    model_pattern = re.search(
                        r'[\s]note[\s]?[0-9]{1,2}\+?[\s]?(ultra)?', name)
                if model_pattern is None:
                    model_pattern = re.search(r'prime[ ]?((plus)|\+)?', name)
                if model_pattern is not None:
                    model = model_pattern.group().replace(' ', '').replace('plus', '+')
                instance_type = 'sim'
            elif 'tv' in name:
                capacity_model = re.search(r'[0-9]{2}[- ]?inch', name)
                if capacity_model is not None:
                    capacity = capacity_model.group()[:2]
                instance_pattern = re.search(r'(hd)|(qled)|(uhd)', name)
                if instance_pattern is not None:
                    instance_type = instance_pattern.group()
                model_pattern = re.search(r'[a-z]{1,2}[0-9]{4}', name)
                if model_pattern is not None:
                    model = model_pattern.group()
            else:
                if instance_type == 'ssd':
                    model_pattern = re.search(r'[\s]t[0-9][\s]', name)
                    if model_pattern is not None:
                        model = model_pattern.group().strip()
                else:
                    model_pattern = re.search(r'(pro)|(evo)', name)
                    if model_pattern is not None:
                        model = model_pattern.group()
                        model_pattern = re.search(r'(\+)|(plus)', name)
                        if model_pattern is not None:
                            model = model + model_pattern.group().replace('plus', '+')
                        if model == 'evo+' and instance_type == '':
                            instance_type = 'microsd'
            for c in colors:
                if c in name:
                    type = c
                    break

        elif brand == 'toshiba':
            model_pattern = re.search(r'[\s\-n][umn][0-9]{3}', name)
            if model_pattern is not None:
                model = model_pattern.group()[1:]
                if instance_type == '':
                    ch = model[0]
                    if ch == 'u':
                        instance_type = 'usb'
                    elif ch == 'n':
                        instance_type = 'sd'
                    elif ch == 'm':
                        instance_type = 'microsd'
            if instance_type == 'usb' and model == '':
                model_pattern = re.search(r'ex[\s-]?ii', name)
                if model_pattern is None:
                    model_pattern = re.search(r'osus', name)
                if model_pattern is not None:
                    model = 'ex'
            if 'transmemory' in name:
                if instance_type == '':
                    instance_type = 'usb'
            if instance_type != 'usb':
                type_pattern = re.search(
                    r'exceria[ ]?((high)|(plus)|(pro))?', name)
                if type_pattern is not None:
                    type = type_pattern.group().replace(' ', '').replace('exceria', 'x')
                elif capacity != '':
                    type_pattern = re.search(
                        r'x[ ]?((high)|(plus)|(pro))?' + capacity[:-2], name)
                    if type_pattern is not None:
                        type = type_pattern.group().replace(' ', '')[
                               :-(len(capacity) - 2)]
                if type == 'xpro' and instance_type == '':
                    instance_type = 'sd'
                if type == 'xhigh' and instance_type == '':
                    instance_type = 'microsd'
            if instance_type == 'usb' and model == '':
                if 'hayaqa' in name or 'hayabusa' in name:
                    model = 'u202'
            if instance_type == 'sd' and model == '':
                model_pattern = re.search(r'silber', name)
                if model_pattern is not None:
                    model = 'n401'
            if instance_type == 'sd' and model == '' and type == '':
                model_pattern = re.search(
                    r'sd[hx]c uhs clas[se] 3 memor(y|(ia)) ((card)|(flash))',
                    name)
                if model_pattern is not None:
                    type = 'xpro'
            if instance_type == 'sd' and model == '' and type == '':
                if 'uhs-ii' in name and 'carte mémoire flash' in name:
                    type = 'xpro'
            if instance_type != 'usb':
                speed_model = re.search(
                    r'[1-9][0-9]{1,2}[\s]?m[bo]/s', name)
                if speed_model is not None:
                    speed = re.search(
                        r'[0-9]{2,3}', speed_model.group()).group()
                    if speed == '260' or speed == '270':
                        if type == '':
                            type = 'xpro'
                    if speed == '90' and type == 'x':
                        if model == '':
                            model = 'n302'
            if 'gb uhs-i (u3 - up to 95mb/s read) flash memory card' in name:
                instance_type = 'microsd'
                type = 'x'
            if 'toshiba pendrive usb high-speed' in name:
                model = 'u202'
            if 'en fnac.es' in name and 'toshiba usb 3.0' in name and 'pendrive / memoria usb' in name:
                model = 'ex'
            if model == 'n101':
                model = ''

        elif brand == 'transcend':
            pass

        identification = ''
        if capacity in ('256gb', '512gb', '1tb', '2tb') and brand not in ('samsung', 'sandisk'):
            identification = brand + '_' + capacity
            continue

        if brand == 'lexar':
            if capacity != '' and type != '' and instance_type != '':
                identification = brand + '_' + capacity + '_' + instance_type + '_' + type

        elif brand == 'sony':
            if (instance_type in ('ssd', 'microsd') or capacity == '1tb') and capacity != '':
                identification = brand + '_' + capacity + '_' + instance_type
            elif instance_type != '' and capacity != '' and type != '':
                identification = brand + '_' + capacity + '_' + instance_type + '_' + type

        elif brand == 'sandisk':
            if capacity != '' and instance_type != '':
                identification = brand + '_' + capacity + '_' + instance_type + '_' + model

        elif brand == 'pny':
            if capacity != '' and instance_type != '':
                identification = brand + '_' + capacity + '_' + instance_type

        elif brand == 'intenso':
            if capacity != '' and type != '':
                identification = brand + '_' + capacity + '_' + type

        elif brand == 'kingston':
            if instance_type != '' and capacity != '':
                identification = brand + '_' + capacity + '_' + instance_type

        elif brand == 'samsung':
            if instance_type in ('microsd', 'ssd', 'sd', 'usb') and capacity != '' and model != '':
                identification = brand + '_' + capacity + '_' + instance_type + '_' + model
            elif instance_type != '' and capacity != '' and type != '' and model != '':
                identification = brand + '_' + capacity + '_' + instance_type + '_' + type + model

        elif brand == 'toshiba':
            if capacity != '' and instance_type != '' and model != '':
                identification = brand + '_' + capacity + '_' + model + '_' + instance_type
            elif capacity != '' and instance_type != '' and type != '':
                identification = brand + '_' + capacity + '_' + type + '_' + instance_type

        elif brand == 'transcend':
            if capacity != '' and instance_type != '':
                identification = brand + '_' + capacity + '_' + instance_type

        result.append({
            'instance_id': id_list[row],
            'brand': brand,
            'price': price,
            'capacity': capacity,
            'instance_type': instance_type,
            'type': type,
            'model': model,
            'item_code': item_code,
            'identification': identification,
            'name': name
        })
    result = pd.DataFrame(result)
    return result
