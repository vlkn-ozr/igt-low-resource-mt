%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  File:     symbols.fst
%  Author:   Ayla Kayabas, Helmut Schmid; YBU, Yildirim Beyazit University, IMS, University of Stuttgart
%  Date:     January 2011
%  Content:  definition of symbol classes
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% All symbols used by the morphology should be defined here


%%% Single Character Symbols %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% lower case consonants
#cons# = bcdfghjklmnpqrstvwxyzşçğ

%upper case consonants
#CONS# = BCDFGHJKLMNPQRSTVWXYZŞÇĞ

% all consonants
#Cons# = #cons# #CONS#


% lower case vowels
#vowel# = aeiouöüıâôû <Y><K>

% upper case vowels
#VOWEL# = AEİOUÖÜIÂÔÛ

% all vowels
#Vowel# = #vowel# #VOWEL#


% lower case letters
#letter# = #cons# #vowel#

% upper case letters
#LETTER# = #CONS# #VOWEL#

% Trigger
#Trigger# = <Q><X><del><DEL><Komp> <passiv><aorist>

% all letters
#Letter# = #Cons# #Vowel# #Trigger# '


%%% Lexicon Entry Markers %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% affix type features
#Affix# = <Prefix><Suffix>

% stem type features (internally used)
#BDKStem# = <BaseStem><DerivStem><CompStem>

% all stem types including the general stem feature <Stem>
% used in the lexicon
#EntryType# = <Stem> #BDKStem# #Affix#


%%% Agreement Features %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% word class features
#WordClass# = <ADJ><ADV><CARD><N><NE><V><OTHER><Suffix>

% stem type feature
#StemType# = <base><deriv><comp>

% classic origin features
#classic# = <free><bound><short><long>

% all origin features
#Origin# = <native><foreign> #classic#

% origin features including the internally used feature <classic>
% which represents the disjunction stored in #classic#
#Origin-cl# = #Origin# <classic>

% complexity features
#Complex# = <simplex><prefderiv><suffderiv>

% inflection class features
#InflClass# = <NomReg><NomReg-p><NomRegCop><NomReg-su><VerbReg><AdjReg-i><AdjReg-i-comp><NEReg><ConjReg>

% all agreement features
#AgrFeat# = #WordClass# #StemType# #Origin-cl#

% all agreement features + inflection class features
#AgrFeatInfl# = #AgrFeat# #InflClass#


%%% Analysis Features %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% number feature
#Number# = <sg><pl>

% gender feature
#Gender# = <masc><fem><neut>

% case feature
#Case# = <nom><acc><gen><dat><loc><abl>

% Person Feature
#Person# = <1><2><3>

% degree feature
#Degree# = <positive><comparative><superlative>

% verbal features
#VerbFeat# = <pres><past><part>

% affix markers
#AFF# = <PREF><SUFF>

% Morphosyntactic Features
#MorphSyn# = #Number# #Gender# #Case# #Person#\
	     #Degree# #VerbFeat# #AFF#


%%% Trigger Features %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% capitalisation feature: lower case, capitalized or fixed
#Cap# = <UC><LC><Cap><Fix>

% Features used to mark the boundaries of morphemes and inflection
#Boundary# = <F>

% all triggers
% <NoDef> marks lexicon entries without default stems
#Trigger# = #Cap# #Boundary# <NoDef>


%%% General Symbol Classes %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#Tag# = #EntryType# #AgrFeatInfl# #MorphSyn# #Trigger#

#AllSym# = #Letter# #Tag#

