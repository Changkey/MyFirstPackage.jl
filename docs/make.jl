using MyFirstPackage
using Documenter

DocMeta.setdocmeta!(MyFirstPackage, :DocTestSetup, :(using MyFirstPackage); recursive=true)

makedocs(;
    modules=[MyFirstPackage],
    authors="Changkey Culing",
    sitename="MyFirstPackage.jl",
    format=Documenter.HTML(;
        canonical="https://changkey.github.io/MyFirstPackage.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/changkey/MyFirstPackage.jl",
    devbranch="main",
)
