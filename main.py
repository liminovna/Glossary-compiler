import argparse

parser = argparse.ArgumentParser(description="""
Эта программа вытащит из книги глоссарий из одного и/или двух слов. 
Пожалуйста, укажите язык, на котором написана книга, передав параметр -l и количество возвращаемых слов/сочетаний с помощью параметра -n.
"""
)

parser.add_argument('-f', "--filename", help='Название файла или путь к нему, если файл лежит не в той же папке, что и программа')
parser.add_argument('-l', "--lang", help='Язык (ru или en)')
parser.add_argument('-n', "--top_n", type=int, nargs='?', default=100, help='Вывести топ-n самых частотных слов/биграмм. По умолчанию 100')
parser.add_argument('-u', "--united", action='store_true', help='По умолчанию составляется два отдельных списка с уни- и биграммами. Чтобы объединить два списка, укажите этот флаг.')

args_ = parser.parse_args()

filename = args_.filename
lang = args_.lang
top_n = args_.top_n
united = args_.united


if __name__ == '__main__':
    assert filename, 'Вы забыли ввести название файла!'

    from classes import GlossaryCompiler

    inst = GlossaryCompiler(filename=filename, lang=lang)
    top_uni, top_bi = inst.get_top(top_n=top_n, united=united, min_threshold=0)
    # top = inst.get_top(top_n=50, united=True, min_threshold=0)
    inst.export()
