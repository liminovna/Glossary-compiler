import argparse

parser = argparse.ArgumentParser(description="""
Эта программа вытащит из книги глоссарий из одного и/или двух слов. 
Пожалуйста, укажите язык, на котором написана книга, передав параметр -l и количество возвращаемых слов/сочетаний с помощью параметра -n.
"""
)

parser.add_argument('-l', "--lang", help='Язык (ru или en)')
parser.add_argument('-n', "--top_n", type=int, nargs='?', default=100, help='Вывести топ-n самых частотных слов/биграмм. По умолчанию 100')
parser.add_argument('-u', "--united", action='store_true', help='По умолчанию составляется два отдельных списка с уни- и биграммами. Чтобы объединить два списка, укажите этот флаг.')

args = parser.parse_args()

opts = args.lang
top_n = args.top_n
united = args.united

print(args)