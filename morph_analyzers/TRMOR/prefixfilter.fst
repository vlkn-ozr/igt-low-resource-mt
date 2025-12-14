%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  File:     prefixfilter.fst
%  Author:   Ayla Kayabas, Helmut Schmid; YBU, Yildirim Beyazit University, IMS, University of Stuttgart
%  Date:     January 2011
%  Content:  enforcement of derivational constraints
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


#include "symbols.fst"


% Check agreement of the word class and origin features
% Delete the agreement features of the prefix

ALPHABET = [#Letter#] <Suffix>

#=wc# = #WordClass#
#=orig# = #Origin#

<Prefix> .* \
[#=wc#]:<> [#=orig#]:<> [#BDKStem#] .* [#=wc#] [#StemType#] [#=orig#] \
[#InflClass#]?
