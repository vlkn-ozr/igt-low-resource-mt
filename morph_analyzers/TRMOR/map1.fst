%**************************************************************************
%  File:     map1.fst  
%  Author:   Ayla Kayabas, Helmut Schmid; YBU, Yildirim Beyazit University, IMS, University of Stuttgart
%  Date:     January 2011
%  Content:  deletes tags in the analysis string
%**************************************************************************

% definition of the symbol classes
#include "symbols.fst"


% delete unwanted symbols in the analysis

ALPHABET = [#Letter# #WordClass# <compound>] \
	<>:[#StemType# #Origin-cl# #InflClass# #Trigger#]

<>:<NoDef>? <>:<Stem> .* |\
<>:<Suffix> <>:[#Complex#] <>:[#WordClass#] .* <SUFF>:<> |\
<>:<Prefix> .* <>:[#WordClass#] .* <PREF>:<>
