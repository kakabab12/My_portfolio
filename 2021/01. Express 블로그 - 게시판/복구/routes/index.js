var express = require('express');
var router = express.Router();

/* GET home page. */
router.get('/', function(req, res, next) {
  Board.find({}, function (err, board) {
      res.render('index', { title: 'Express', board: board });
  });
});

module.exports = router;
