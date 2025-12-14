%**************************************************************************
%  File:     map2.fst  
%  Author:   Ayla Kayabas, Helmut Schmid; YBU, Yildirim Beyazit University, IMS, University of Stuttgart
%  Date:     January 2011
%  Content:  deletes tags on the surface string
%**************************************************************************

% definition of the symbol classes
#include "symbols.fst"


% delete unwanted symbols on the "surface"
% and map the feature <Stems> to the more specific features
% <BaseStem> <DerivStem> and <CompStem>

ALPHABET = [#Letter# #WordClass# #StemType# #Origin# #Complex# #InflClass#] \
	<classic>:[#classic#]

[#Affix#] .* |\
<NoDef>? (<Stem>:<BaseStem>  .* <base>  |\
	  <Stem>:<DerivStem> .* <deriv> |\
	  <Stem>:<CompStem>  .* <comp>) .*
