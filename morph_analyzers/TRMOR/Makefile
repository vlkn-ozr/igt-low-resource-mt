
SOURCES = morph.fst symbols.fst map1.fst map2.fst defaultstems.fst \
	suffixfilter.fst prefixfilter.fst compoundfilter.fst inflection.fst \
	inflectionfilter.fst phon.fst

.PHONY: targets

targets: morph.ca

morph.a: map1.a map2.a prefixfilter.a suffixfilter.a compoundfilter.a \
	inflection.a inflectionfilter.a lexicon

test.a: map1.a map2.a prefixfilter.a suffixfilter.a compoundfilter.a \
	inflection.a inflectionfilter.a lexicon lex

%.a: %.fst
	fst-compiler-utf8 $< $@

%.ca: %.a
	fst-compact $< $@

archive:
	gtar -zcf TRMOR-`date '+%y%m%d'`.tar.gz Makefile lexicon *.fst

clean:
	-rm *.a *~ Makefile.bak 2>&- > /dev/null


# DO NOT DELETE

morph.a: symbols.fst defaultstems.fst phon.fst
test.a: symbols.fst defaultstems.fst phon.fst
map1.a: symbols.fst
map2.a: symbols.fst
suffixfilter.a: symbols.fst
prefixfilter.a: symbols.fst
compoundfilter.a: symbols.fst
inflection.a: symbols.fst
inflectionfilter.a: symbols.fst
phon.a: symbols.fst
