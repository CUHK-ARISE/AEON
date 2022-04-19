import argparse
import re


CHAR_ENTITIES = {'nbsp': ' ', 'lt': '<', 'gt': '>', 'amp': '&', 'quot': '"', 'apos': '\'',
                 'dollar': '$', 'cent': '¢', 'pound': '£', 'yen': '¥', 'euro': '€',
                 'num': '#', 'percnt': '%', 'ast': '*'}


def ReplaceCharEntity(htmlstr):
    re_charEntity = re.compile(r'(\s#|&#|&)(?P<name>\w+);')
    search_char = re_charEntity.search(htmlstr)
    
    while search_char:
        entity = search_char.group()
        key = search_char.group('name')
        try:
            # Convert HTML codes to characters
            htmlstr = re_charEntity.sub(chr(int(key)), htmlstr, 1)
        except ValueError:
            try:
                # Convert HTML entity names to characters
                htmlstr = re_charEntity.sub(CHAR_ENTITIES[key], htmlstr, 1)
            except KeyError:
                # Preserve & = and
                print('HTML Char Entity: %s' % key)
                print(htmlstr)
                htmlstr = re_charEntity.sub(' and ' + key, htmlstr, 1)
        
        search_char = re_charEntity.search(htmlstr)
    
    return htmlstr


def FilterTag(htmlstr):
    htmlstr = ReplaceCharEntity(htmlstr)
    for k in CHAR_ENTITIES.keys():
        htmlstr = htmlstr.replace(' %s;' % k, CHAR_ENTITIES[k])
    
    re_cdata = re.compile('//<!\[CDATA\[[^>]*//\]\]>', re.I)
    htmlstr = re_cdata.sub('', htmlstr)
    
    re_script = re.compile('<\s*script[^>]*>[^<]*<\s*/\s*script\s*>', re.I)
    htmlstr = re_script.sub('', htmlstr)
    
    re_style = re.compile('<\s*style[^>]*>[^<]*<\s*/\s*style\s*>', re.I)
    htmlstr = re_style.sub('', htmlstr)
    
    re_br = re.compile('<br\s*?/?>')
    htmlstr = re_br.sub(' ', htmlstr)
    
    re_h = re.compile('</?\w+[^>]*>')
    htmlstr = re_h.sub('', htmlstr)
    
    re_comment = re.compile('<!--[^>]*-->')
    htmlstr = re_comment.sub('', htmlstr)
    
    re_blankline = re.compile('\n+')
    htmlstr = re_blankline.sub(' ', htmlstr)
    
    re_http = re.compile('[http|https]*://[a-zA-Z0-9.?/&=:]*', re.S)
    htmlstr = re_http.sub('', htmlstr)
    
    re_fucking = re.compile('(f|F)(\$|#|@|!|\^|&|\*){6,}')
    htmlstr = re_fucking.sub('fucking', htmlstr)
    
    re_fuck = re.compile('(f|F)(\$|#|@|!|\^|&|\*)+')
    htmlstr = re_fuck.sub('fuck', htmlstr)
    
    return htmlstr


parser = argparse.ArgumentParser()
parser.add_argument('--txt', type=str, default='')
args = parser.parse_args()


with open(args.txt) as f:
    raw = f.read()
    for i in range(161):
        if (i < 32 and i != 10 and i != 9) or i > 127:
            raw = raw.replace(chr(i), ' ')
    raw = raw.splitlines()

for i in range(len(raw)):
    raw[i] = FilterTag(raw[i])
    
    search_u = raw[i].find('\\u')
    while search_u >= 0:
        key = raw[i][search_u:search_u + 6]
        try:
            raw[i] = raw[i].replace(key, chr(int(key[2:], 16)))
            search_u = raw[i].find('\\u')
        except ValueError:
            raw[i] = raw[i].replace(key, ' ' + key[1:])
            search_u = raw[i].find('\\u')
    
    raw[i] = raw[i].replace('\\n', ' ')
    raw[i] = raw[i].replace('\\$', '$')
    raw[i] = raw[i].replace('\\""', '"')
    raw[i] = raw[i].replace('\\', ' ')

with open(args.txt, 'w') as f:
    f.write('\n'.join(raw))
