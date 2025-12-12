import { ModeToggle } from "@/components/ModeToggle";
import { Button } from "@/components/ui/button";
import {
  Sheet,
  SheetContent,
  SheetTrigger,
  SheetTitle
} from "@/components/ui/sheet";
import { Menu } from "lucide-react";
import { useState } from "react";
import { Link } from "react-router-dom";

function Navbar() {
  const [isOpen, setIsOpen] = useState(false);

  const links = [
    { name: "Dashboard", path: "/dashboard" },
    { name: "Read Articles", path: "/read-articles" },
    { name: "Label Articles", path: "/label-articles" },
    { name: "Solutions", path: "/solutions" },
    { name: "Methodology", path: "/methodology" }
  ];

  return (
    <nav className="bg-background border-b">
      <div className="flex h-16 items-center justify-between px-4">
        <Link to="/" className="hover:text-primary text-lg font-bold">
          BERTMAN
        </Link>

        {/* Desktop Menu */}
        <div className="hidden items-center space-x-8 md:flex">
          {links.map((link) => (
            <Link
              key={link.path}
              to={link.path}
              className="hover:text-primary text-sm font-medium transition-colors">
              {link.name}
            </Link>
          ))}
          <ModeToggle />
        </div>

        {/* Mobile Menu */}
        <div className="md:hidden">
          <Sheet open={isOpen} onOpenChange={setIsOpen}>
            <SheetTrigger asChild={true}>
              <Button variant="ghost" size="icon">
                <Menu className="h-6 w-6" />
                <span className="sr-only">Toggle menu</span>
              </Button>
            </SheetTrigger>
            <SheetContent side="right">
              <SheetTitle className="sr-only">Menu</SheetTitle>
              <div className="mt-4 flex flex-col space-y-4">
                {links.map((link) => (
                  <Link
                    key={link.path}
                    to={link.path}
                    className="hover:text-primary text-sm font-medium transition-colors"
                    onClick={() => setIsOpen(false)}>
                    {link.name}
                  </Link>
                ))}
                <div className="flex justify-start pt-4">
                  <ModeToggle />
                </div>
              </div>
            </SheetContent>
          </Sheet>
        </div>
      </div>
    </nav>
  );
}

export default Navbar;
