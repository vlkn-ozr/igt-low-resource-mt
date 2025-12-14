%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  File:     inflectionfilter.fst
%  Author:   Ayla Kayabas, Helmut Schmid; YBU, Yildirim Beyazit University, IMS, University of Stuttgart
%  Date:     January 2011
%  Content:  definition of the inflectional filter 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% definition of the symbol classes
#include "symbols.fst"


% The following inflection filter ensures that the base stems
% are combined with the correct inflectional endings

ALPHABET = [#Letter# #EntryType# #WordClass# #Boundary#]


$=1$ = [#InflClass#]:<>

.* $=1$ $=1$ .* [#Cap#]
