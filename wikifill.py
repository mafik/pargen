# -*- coding: utf-8 -*-
import fileinput
import re
text = False
art = ''
for line in fileinput.input():
  if re.search('<text ', line):
    text = True
  if re.search('#redirect', line, re.I):
    text = False
  if text:
    art += ' ' + line
    if re.search('<\/text>', line):
      text = False
      art = art.decode('utf-8')
      art = re.sub('<.*>', '', art)
      art = re.sub('&amp;', '&', art)
      art = re.sub('&lt;', '<', art)
      art = re.sub('&gt;', '>', art)
      art = re.sub('<ref[^<]*<\/ref>', '', art)
      art = re.sub('<timeline.*</timeline>', '', art, flags=re.DOTALL)
      art = re.sub('<[^>]*>', '', art)
      art = re.sub('\[http:[^] ]*', '[', art)
      art = re.sub('\|thumb', '', art, flags=re.I)
      art = re.sub('\|left', '', art, flags=re.I)
      art = re.sub('\|right', '', art, flags=re.I)
      art = re.sub('\|\d+px', '', art, flags=re.I)
      art = re.sub('\[\[image:[^\[\]]*\|', '', art, flags=re.I)
      art = re.sub('\[\[category:([^|\]]*)[^]]*\]\]', '\\1', art, flags=re.I)
      art = re.sub('\[\[kategoria:([^|\]]*)[^]]*\]\]', '\\1', art, flags=re.I)
      art = re.sub('\[\[[a-z\-]*:[^\]]*\]\]', '', art)
      art = re.sub('\[\[[^\|\]]*\|', '[[', art)
      art = re.sub('{{[^{}]*}}', '', art)
      art = re.sub('{{[^{}]*}}', '', art)
      art = re.sub('{[^}]*}', '', art)
      art = re.sub('\[', '', art)
      art = re.sub('\]', '', art)
      art = re.sub('&[^;]*;', ' ', art)
      art = re.sub('http(s)?:[^ ]*', ' ', art)
      art = art.replace('0', ' zero ')
      art = art.replace('1', ' jeden ')
      art = art.replace('2', ' dwa ')
      art = art.replace('3', ' trzy ')
      art = art.replace('4', ' cztery ')
      art = art.replace('5', u' pięć ')
      art = art.replace('6', u' sześć ')
      art = art.replace('7', ' siedem ')
      art = art.replace('8', ' osiem ')
      art = art.replace('9', u' dziewięć ')
      art = re.sub(u'== (Bibliografia|Zobacz też|Linki zewnętrzne) ==.*$', '', art, flags=re.DOTALL|re.I)
      art = re.sub(u'\W+', ' ', art.lower(), flags=re.UNICODE).strip() # interpunkcja
      print(art.encode('utf-8'))
      art = ''
