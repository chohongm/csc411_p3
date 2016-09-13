<?php

function db_connect($db_name = 'statqueue') {
   #domain name for website: 'ab18051_statq', otherwise, localhost.
   #db name: 'ab18051_statqueue', if not, statqueue
   @ $result = new mysqli('localhost', 'statqueue', 'Tbalftwf89', $db_name);
   if (!$result) {
     throw new Exception('Could not connect to database server');
   } else {
     return $result;
   }
}



?>
