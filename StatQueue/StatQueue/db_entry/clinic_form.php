<?php
$DOCUMENT_ROOT = $_SERVER['DOCUMENT_ROOT'];
require_once("$DOCUMENT_ROOT/functions/statq_fns.php");
do_header();

if (check_valid_user() == 'master'){
	display_clinic_form();
} else {
	echo "<br /><br /><p>Sorry, add new clinic option is only accessible to the webmasters.
	  <br /><br />";
}
do_footer();
?>