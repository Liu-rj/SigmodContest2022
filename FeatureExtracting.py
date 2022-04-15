import pandas as pd
import re


def extract_x2(data: pd.DataFrame) -> pd.DataFrame:
    """Clean X2.csv data to a readable format.

    :param data: X2.csv

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

         if the value can't extract from the information given, '0' will be filled.
    """
    brands = ['sandisk', 'lexar', 'kingston', 'intenso', 'toshiba', 'sony', 'pny', 'samsung']
    series = {'sandisk': ['extreme', 'cruzer', 'ultra', 'traveler', 'sdhc', 'usb', 'adapt'],
              'lexar': ['ultra', 'jumpdrive'],
              'toshiba': ['exceria', 'traveler', 'sdhc', 'memoria'],
              'kingston': ['traveler'],
              'sony': ['USM32GQX'],
              'intenso': ['cs/ultra', 'premium', 'ultra', 'micro'],
              'pny': [],
              'samsung': []
              }

    intenso_type = ["basic", "rainbow", "high speed", "speed", "premium", "alu", "business", "micro",
                    "imobile", "cmobile", "mini", "ultra", "slim", "flash", "mobile"]

    colors = ['midnight black', 'prism white', 'prism black', 'prism green', 'prism blue', 'canary yellow',
              'flamingo pink', 'cardinal red', 'smoke blue', 'deep blue', 'coral orange',
              'black sky', 'gold sand', 'blue mist and peach cloud', 'orchid gray',
              'metallic copper', 'lavender purple', 'ocean blue', 'pure white', 'alpine white',
              'copper', 'red', 'black', 'blue', 'white', 'silver', 'gold', 'violet', 'purple',
              'brown', 'orange', 'coral', 'pink']

    result = []

    for row in range(data.shape[0]):
        name_info = data['name'][row]

        size = '0'
        mem_type = '0'
        brand = '0'
        product_type = '0'
        model = '0'
        item_code = '0'
        series_name = '0'

        size_model = re.search(r'[0-9]{1,4}[ ]*[gt][bo]', name_info)
        if size_model is not None:
            size = size_model.group()[:].replace(' ', '')

        for b in brands:
            if b in name_info:
                brand = b
                for sn in series[b]:
                    if sn in name_info:
                        series_name = sn
                        break
                break

        mem_model = re.search(r'ssd', name_info)
        if mem_model is None:
            mem_model = re.search(r'micro[- ]?sd[hx]?c?', name_info)
        if mem_model is None:
            mem_model = re.search(r'sd[hx]c', name_info)
        if mem_model is None:
            mem_model = re.search(r'usb', name_info)
        if mem_model is None:
            mem_model = re.search(r'sd(?!cz)', name_info)
        if mem_model is None:
            mem_model = re.search(r'secure digital', name_info)
        if mem_model is None:
            mem_model = re.search(r'xqd', name_info)
        if mem_model is None:
            mem_model = re.search(r'ljd', name_info)
        if mem_model is not None:
            mem_type = mem_model.group()
            if mem_type not in ('ssd', 'usb', 'xqd'):
                if 'micro' in mem_type:
                    mem_type = 'microsd'
                elif 'ljd' in mem_type:
                    mem_type = 'usb'
                else:
                    mem_type = 'sd'
        if 'adapt' in name_info and mem_type == '0':
            mem_type = 'microsd'

        item_code_model = re.search(r'\((mk)?[0-9]{6,10}\)', name_info)
        if item_code_model is not None:
            item_code = item_code_model.group()[1:-1]

        if brand == "intenso":
            model_model = re.search(r'[0-9]{7}', name_info)
            if model_model is not None:
                model = model_model.group()[:]

            type_model = re.search(r'(high\s)?[a-z]+\s(?=line)', name_info)
            if type_model is not None:
                product_type = type_model.group()[:].replace(' ', '')
            else:
                for t in intenso_type:
                    if t in name_info:
                        product_type = t.replace(' ', '')
                        break

        elif brand == "lexar":
            type_model = re.search(
                r'((jd)|[\s])[a-wy-z][0-9]{2}[a-z]?', name_info)
            if type_model is None:
                type_model = re.search(r'[\s][0-9]+x(?![a-z0-9])', name_info)
            if type_model is None:
                type_model = re.search(r'(([\s][x])|(beu))[0-9]+', name_info)
            if type_model is not None:
                product_type = type_model.group().strip() \
                    .replace('x', '').replace('l', '').replace('j', '').replace('d', '') \
                    .replace('b', '').replace('e', '').replace('u', '')

            if mem_type == '0':
                if 'drive' in name_info:
                    mem_type = 'usb'
            if 'lexar 8gb jumpdrive v10 8gb usb 2.0 tipo-a blu unità flash usb' in name_info:
                product_type = 'c20c'

        elif brand == 'sony':
            if mem_type == '0':
                if 'ux' in name_info or 'uy' in name_info or 'sr' in name_info:
                    mem_type = 'microsd'
                elif 'uf' in name_info:
                    mem_type = 'sd'
                elif 'usm' in name_info or size == '1tb':
                    mem_type = 'usb'

            type_model = re.search(r'((sf)|(usm))[-]?[0-9a-z]{1,6}', name_info)
            if type_model is not None:
                product_type = type_model.group().replace('-', '').replace('g', '')
                for c in range(ord('0'), ord('9')):
                    product_type = product_type.replace(chr(c), '')
                if 'sf' in product_type and mem_type == '0':
                    mem_type = 'sd'
                product_type = product_type.replace('sf', '').replace('usm', '')
            elif mem_type in ('sd', 'usb'):
                if 'machqx' in name_info:
                    product_type = 'qx'
                elif 'type-c' in name_info or 'type c' in name_info:
                    product_type = 'ca'
                type_model = re.search(
                    r'(serie[s]?[\s-]?[a-z]{1,2}[\s])|([\s][a-z]{1,2}[\-]?serie[s]?)', name_info)
                if type_model is not None:
                    product_type = type_model.group().replace(
                        ' ',
                        '').replace(
                        '-',
                        '').replace(
                        'g',
                        '')
                    product_type = product_type.replace('series', '').replace('serie', '')

        elif brand == 'sandisk':
            model_model = re.search(r'ext.*(\s)?((plus)|(pro)|\+)', name_info)
            if model_model is not None:
                model = 'ext+'
            else:
                model_model = re.search(r'ext(reme)?', name_info)
                if model_model is not None:
                    model = 'ext'
                else:
                    model_model = re.search(r'fit', name_info)
                    if model_model is None:
                        model_model = re.search(r'glide', name_info)
                    if model_model is None:
                        model_model = re.search(r'blade', name_info)
                    if model_model is not None:
                        model = model_model.group()
                    else:
                        model_model = re.search(
                            r'ultra(\s)?((plus)|(pro)|\+|(performance)|(android))', name_info)
                        if model_model is None:
                            model_model = re.search(
                                r'sandisk 8gb ultra sdhc memory card, class 10, read speed up to 80 mb/s \+ sd adapter',
                                name_info)
                        if model_model is None:
                            model_model = re.search(
                                r'sandisk sdhc [0-9]+gb 80mb/s cl10\\n', name_info)
                        if model_model is not None:
                            model = 'ultra+'
                        else:
                            model_model = re.search(r'ultra', name_info)
                            if model_model is not None:
                                model = 'ultra'
                            else:
                                model_model = re.search(r'dual', name_info)
                                if model_model is None:
                                    model_model = re.search(
                                        r'double connect.*', name_info)
                                if model_model is not None:
                                    model = 'ultra'

            if 'accessoires montres' in name_info:
                if 'extreme' in name_info:
                    mem_type = 'microsd'
                    model = 'ultra+'
                elif 'ext pro' in name_info:
                    mem_type = 'microsd'
                    model = 'ext+'
            if 'adapter' in name_info or 'adaptateur' in name_info:
                mem_type = 'microsd'
            if mem_type == '0':
                if 'drive' in name_info:
                    mem_type = 'usb'
                elif 'cruzer' in name_info:
                    mem_type = 'usb'
                elif model in ('glide', 'fit'):
                    mem_type = 'usb'
            if 'sandisk - ' + size + ' extreme en fnac.es' in name_info:
                mem_type = 'usb'

        elif brand == 'pny':
            type_model = re.search(r'att.*?[3-4]', name_info)
            if type_model is not None:
                product_type = type_model.group().replace(' ', '').replace('-', '')
                product_type = 'att' + \
                               list(filter(lambda ch: ch in '0123456789', product_type))[0]
                if mem_type == '0':
                    mem_type = 'usb'

        elif brand == 'kingston':
            if mem_type == '0':
                if 'savage' in name_info or 'hx' in name_info or 'hyperx' in name_info:
                    mem_type = 'usb'
                elif 'ultimate' in name_info:
                    mem_type = 'sd'
            model_model = re.search(
                r'(dt[i1]0?1?)|(data[ ]?t?travel?ler)', name_info)
            if model_model is not None:
                model = 'data traveler'
                type_model = re.search(r'(g[234])|(gen[ ]?[234])', name_info)
                if type_model is not None:
                    product_type = type_model.group()[-1:]
            else:
                type_model = re.search(
                    r'[\s]((g[234])|(gen[ ]?[234]))[\s]', name_info)
                if type_model is not None:
                    product_type = type_model.group().strip()[-1:]
                    model = 'data traveler'
            if model == 'data traveler' and mem_type == '0':
                mem_type = 'usb'

        elif brand == 'samsung':
            if 'lte' in name_info:
                model_model = re.search(
                    r'[\s][a-z][0-9]{1,2}[a-z]?[\s]((plus)|\+)?', name_info)
                if model_model is None:
                    model_model = re.search(
                        r'[\s]note[\s]?[0-9]{1,2}\+?[\s]?(ultra)?', name_info)
                if model_model is None:
                    model_model = re.search(r'prime[ ]?((plus)|\+)?', name_info)
                if model_model is not None:
                    model = model_model.group().replace(' ', '').replace('plus', '+')
                mem_type = 'sim'
            elif 'tv' in name_info:
                size_model = re.search(r'[0-9]{2}[- ]?inch', name_info)
                if size_model is not None:
                    size = size_model.group()[:2]
                mem_model = re.search(r'(hd)|(qled)|(uhd)', name_info)
                if mem_model is not None:
                    mem_type = mem_model.group()
                model_model = re.search(r'[a-z]{1,2}[0-9]{4}', name_info)
                if model_model is not None:
                    model = model_model.group()
            else:
                if mem_type == 'ssd':
                    model_model = re.search(r'[\s]t[0-9][\s]', name_info)
                    if model_model is not None:
                        model = model_model.group().strip()
                else:
                    model_model = re.search(r'(pro)|(evo)', name_info)
                    if model_model is not None:
                        model = model_model.group()
                        model_model = re.search(r'(\+)|(plus)', name_info)
                        if model_model is not None:
                            model = model + model_model.group().replace('plus', '+')
                        if model == 'evo+' and mem_type == '0':
                            mem_type = 'microsd'
            for c in colors:
                if c in name_info:
                    product_type = c
                    break

        elif brand == 'toshiba':
            model_model = re.search(r'[\s\-n][umn][0-9]{3}', name_info)
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
                model_model = re.search(r'ex[\s-]?ii', name_info)
                if model_model is None:
                    model_model = re.search(r'osus', name_info)
                if model_model is not None:
                    model = 'ex'
            if 'transmemory' in name_info:
                if mem_type == '0':
                    mem_type = 'usb'
            if mem_type != 'usb':
                type_model = re.search(
                    r'exceria[ ]?((high)|(plus)|(pro))?', name_info)
                if type_model is not None:
                    product_type = type_model.group().replace(' ', '').replace('exceria', 'x')
                elif size != '0':
                    type_model = re.search(
                        r'x[ ]?((high)|(plus)|(pro))?' + size[:-2], name_info)
                    if type_model is not None:
                        product_type = type_model.group().replace(' ', '')[
                                       :-(len(size) - 2)]
                if product_type == 'xpro' and mem_type == '0':
                    mem_type = 'sd'
                if product_type == 'xhigh' and mem_type == '0':
                    mem_type = 'microsd'
            if mem_type == 'usb' and model == '0':
                if 'hayaqa' in name_info or 'hayabusa' in name_info:
                    model = 'u202'
            if mem_type == 'sd' and model == '0':
                model_model = re.search(r'silber', name_info)
                if model_model is not None:
                    model = 'n401'
            if mem_type == 'sd' and model == '0' and product_type == '0':
                model_model = re.search(
                    r'sd[hx]c uhs clas[se] 3 memor(y|(ia)) ((card)|(flash))',
                    name_info)
                if model_model is not None:
                    product_type = 'xpro'
            if mem_type == 'sd' and model == '0' and product_type == '0':
                if 'uhs-ii' in name_info and 'carte mémoire flash' in name_info:
                    product_type = 'xpro'
            if mem_type != 'usb':
                speed_model = re.search(
                    r'[1-9][0-9]{1,2}[\s]?m[bo]/s', name_info)
                if speed_model is not None:
                    speed = re.search(
                        r'[0-9]{2,3}', speed_model.group()).group()
                    if speed == '260' or speed == '270':
                        if product_type == '0':
                            product_type = 'xpro'
                    if speed == '90' and product_type == 'x':
                        if model == '0':
                            model = 'n302'
            if 'gb uhs-i (u3 - up to 95mb/s read) flash memory card' in name_info:
                mem_type = 'microsd'
                product_type = 'x'
            if 'toshiba pendrive usb high-speed' in name_info:
                model = 'u202'
            if 'en fnac.es' in name_info and 'toshiba usb 3.0' in name_info and 'pendrive / memoria usb' in name_info:
                model = 'ex'
            if model == 'n101':
                model = '0'

        elif brand == 'transcend':
            pass

        result.append([
            data['id'][row],
            brand,
            size,
            mem_type,
            product_type,
            model,
            item_code,
            series_name,
            name_info
        ])

    result = pd.DataFrame(result,
                          columns=['id', 'brand', 'capacity', 'mem_type', 'type', 'model', 'item_code', 'series',
                                   'name'])
    return result
