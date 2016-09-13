<?php
  $DOCUMENT_ROOT = $_SERVER['DOCUMENT_ROOT'];
  require_once("$DOCUMENT_ROOT/functions/statq_fns.php");
  do_header();
  
  @ $db = db_connect();

  if (isset($_SESSION['master'])){
  	if (!filled_out($_POST)){
  		throw new Exception("You have not entered all the required details.<br />"
  				."Please go back and try again.");
	} else if(clinic::check_clinic($db, $_POST['clinicname'])){
		throw new Exception("The clinic you are trying to register already exist in our database.");
	} else {	  	
	  	if (!get_magic_quotes_gpc()) {
	  		foreach ($_POST as $value){
	  			$value = addslashes($value);
	  		}
	  	}

	  	echo "<h2>Touch-in Clinic Entry Results</h2>";
	  	clinic::register_clinic($db, $_POST);
	  	
	  	$db->close();
	  	do_footer();
	  }
  }
?>