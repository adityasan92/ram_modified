"use strict";

var fs = require('fs');
var graphLength = 150000;


function parseFile(fileName, cb){

    fs.readFile(folderName + fileName, 'utf8', function (err,data) {
        if (err) {
            return console.log(err);
        }
        var linesParser = data.split(']]');
        //console.log(res);
        //console.log(linesParser);
        var x = [];
        var y = [];
        console.log(linesParser[0]);
        for(var i =0; i<linesParser.length;i++){
            linesParser[i] = linesParser[i] + "]]"
            //console.log(typeof(linesParser[i]));
            //console.log(linesParser[i]);
            var datapointParser = linesParser[i].split(',');
            //console.log(datapointParser[1]);
            if(Number(datapointParser[0]) > 150000){
                break;
            }
            datapointParser[1] = datapointParser[1].split(' ').join(',')//datapointParser[1].replace("  ", ",")
            //console.log(datapointParser[1])
            datapointParser[1] = datapointParser[1].split(',,,,').join(',')
            datapointParser[1] = datapointParser[1].split(',,,').join(',')
            datapointParser[1] = datapointParser[1].split(',,').join(',')
            datapointParser[1] = datapointParser[1].split(' -').join(',-')
            datapointParser[1] = datapointParser[1].split('[,').join('[')
            datapointParser[1] = datapointParser[1].split(',]').join(']')
            var array = JSON.parse(datapointParser[1]);
            if(Number(datapointParser[0]) != 0 && Number(datapointParser[1]) !=0){
                 x.push(Number(datapointParser[0]));
                y.push(array);
            }
           
        }
        cb(x,y);

    });

}

function checker(files){
    if(counter > files.length - 1  ){
        console.log(dataMap); 
        var json = JSON.stringify(dataMap);
        console.log(dataMap[0]["y"])
        fs.writeFile('concreteDropout_input.json', json, 'utf8');
    }
}
var folderName = './ConcreteDropout_Reward/inputProbablity/';
var files = fs.readdirSync(folderName);
console.log(files);
var counter = 0; 
var dataMap = {};
for(let i=0; i<files.length;i++){
    dataMap[i] = {};
    parseFile(files[i], (x, y)=>{
        dataMap[Number(i)]["x"] =  x;
        dataMap[Number(i)]["y"] =  y;
        counter = counter +1
        checker(files);
    })
}
