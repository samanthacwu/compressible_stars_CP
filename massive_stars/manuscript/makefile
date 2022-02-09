.PHONY: ms_MassiveStarWaves.pdf all clean

all: ms_MassiveStarWaves.pdf

ms_MassiveStarWaves.pdf: ms_MassiveStarWaves.tex intro.tex music.tex simulation_details.tex results_damping.tex results_surface.tex transfer_verification.tex transfer_extrapolation.tex conclusions.tex biblio.bib appendix_mesa_dedalus.tex      appendix_numerical_methods.tex appendix_transforms.tex appendix_mesa.tex
	latexmk -pdf -pdflatex="pdflatex -interaction=nonstopmode" -use-make ms_MassiveStarWaves.tex

clean:
	latexmk -CA
